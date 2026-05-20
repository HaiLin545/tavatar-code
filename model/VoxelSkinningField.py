import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import knn_points
from utils.vis import weights_to_rgb, save_ply_with_color


class VoxelSkinningField(nn.Module):
    def __init__(self, canonical_verts, canonical_weights, resolution=64, padding=0.1):
        """
        canonical_verts: (N, 3) SMPL 标准 T-pose 顶点
        canonical_weights: (N, 24) SMPL 原始权重
        resolution: 体素分辨率，越高越平滑，建议 64 或 128
        padding: Bounding Box 的外扩比例
        """
        super().__init__()

        # 1. 定义 Bounding Box
        min_xyz = canonical_verts.min(dim=0)[0] - padding
        max_xyz = canonical_verts.max(dim=0)[0] + padding
        self.register_buffer("min_xyz", min_xyz)
        self.register_buffer("max_xyz", max_xyz)

        # 2. 构建体素坐标网格
        # 生成 x, y, z 的线性空间
        x = torch.linspace(min_xyz[0], max_xyz[0], resolution)
        y = torch.linspace(min_xyz[1], max_xyz[1], resolution)
        z = torch.linspace(min_xyz[2], max_xyz[2], resolution)

        # 生成 grid (D, H, W, 3) -> (Res, Res, Res, 3)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
        voxel_centers = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(
            canonical_verts.device
        )

        # 展平以便通过 KNN 搜索
        flat_voxel_centers = voxel_centers.reshape(1, -1, 3)  # (1, Res^3, 3)
        ref_verts = canonical_verts.unsqueeze(0)  # (1, N, 3)

        print("Baking Skinning Field... (This runs only once)")

        # 3. 最近邻搜索 (Nearest Neighbor Search)
        # 对于每个体素中心，找到最近的 SMPL 顶点索引
        # 使用 pytorch3d 的 knn_points，或者 scipy cKDTree (如果在 cpu 上做)
        # 这里假设在 GPU 上用 pytorch3d
        knn = knn_points(flat_voxel_centers, ref_verts, K=1)
        nearest_idx = knn.idx[0, :, 0]  # (Res^3, )

        # 4. 赋值权重
        # 根据索引取权重
        baked_weights = canonical_weights[nearest_idx]  # (Res^3, 24)

        # 5. 转换回 Grid 形状 (1, 24, Res, Res, Res) 适配 grid_sample
        # 注意 grid_sample 需要通道在第2维
        baked_weights = baked_weights.reshape(resolution, resolution, resolution, 24)
        baked_weights = baked_weights.permute(3, 0, 1, 2).unsqueeze(0)

        # 注册为 buffer，不参与梯度更新，但随模型保存
        self.register_buffer("volume_weights", baked_weights)

        # visulize this volume
        self._visualize_weights(flat_voxel_centers)

    def _visualize_weights(self, flat_voxel_centers):
        print("Visualizing Skinning Field...")
        with torch.no_grad():
            # (1, 24, D, H, W) -> (D*H*W, 24)
            weights_flat = (
                self.volume_weights.squeeze(0).permute(1, 2, 3, 0).reshape(-1, 24)
            )

            pca_rgb = weights_to_rgb(weights_flat)

            # Save to PLY
            points = flat_voxel_centers.squeeze(0)
            
            save_ply_with_color("debug_voxel_skinning.ply", points, pca_rgb)

    def forward(self, shaped_vertices):
        """·
        shaped_vertices: (B, N, 3) 加上 offset 后的顶点
        """
        B, N, _ = shaped_vertices.shape

        # 1. 坐标归一化到 [-1, 1] 用于 grid_sample
        # formula: 2 * (x - min) / (max - min) - 1
        dims = self.max_xyz - self.min_xyz
        norm_coords = 2.0 * (shaped_vertices - self.min_xyz) / dims - 1.0

        # 调整形状适配 grid_sample: (B, 1, 1, N, 3)
        # grid_sample 通常处理 4D (N, C, H, W) 或 5D (N, C, D, H, W)
        # 我们把所有顶点看作是 "D" 维度的一行采样点
        sample_grid = norm_coords.view(B, 1, 1, N, 3)

        # 2. 三线性插值采样
        # volume_weights: (1, 24, Res, Res, Res) -> 广播到 B
        weights_out = F.grid_sample(
            self.volume_weights.expand(B, -1, -1, -1, -1),
            sample_grid,
            align_corners=True,
            mode="bilinear",  # 3D input 下其实是 trilinear
            padding_mode="border",  # 超出范围取边界值，这很重要
        )

        # weights_out shape: (B, 24, 1, 1, N)
        weights_out = weights_out.view(B, 24, N).permute(0, 2, 1)  # (B, N, 24)

        return weights_out

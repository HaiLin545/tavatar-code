import lightning as L
import torch
import os
import toml
from model.deformer import HumanDeformer
from gaussian import render
from utils.loss import l1_loss, ssim
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    mesh_normal_consistency,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)
from model.equilateral import compute_equilateral_loss_for_meshes
from utils.eval import Evaluator, calculate_chamfer_p2s
from torchvision.utils import save_image, make_grid
import imageio.v3 as iio
from pytorch3d.io import IO


class TavatarModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.deformer = HumanDeformer(cfg)
        self.evaluator = Evaluator()

        # import lpips
        # self.lpips_fn = lpips.LPIPS(net="alex").cuda()
        # self.lpips_fn.require_grad = False
        self.training_step_outputs = None
        self.pred_step_outputs = []
        self.test_step_outputs = []

    def on_test_start(self):
        self.test_dir = os.path.join(self.logger.log_dir, "test")
        os.makedirs(self.test_dir, exist_ok=True)

    def on_fit_start(self):
        log_dir = self.logger.log_dir
        with open(os.path.join(log_dir, "config.toml"), "w") as f:
            toml.dump(self.cfg, f)
        self.render_path = os.path.join(log_dir, "render")
        os.makedirs(self.render_path, exist_ok=True)

    def training_step(self, batch, batch_idx):
        smpl_params = batch["smpl_params"]
        camera_params = batch["camera_params"]

        if self.cfg.dataset.random_bg:
            camera_params["bg"] = batch["bgcolor"]
        camera_param = {k: v[0] for k, v in camera_params.items()}

        gaussians = self.deformer(smpl_params)
        rendered = render(self.cfg, gaussians, camera_param)

        loss_dict = self.compute_loss(batch, gaussians, rendered)
        loss = loss_dict["total_loss"]

        for key in loss_dict:
            self.log(f"train/{key}", loss_dict[key], on_step=False, on_epoch=True)

        self.training_step_outputs = (batch, gaussians, rendered)
        return loss

    @torch.inference_mode()
    def on_train_epoch_end(self):
        batch, gaussians, rendered = self.training_step_outputs
        # rgb_pred = rendered["image"].permute(1, 2, 0).cpu()
        rgb_pred = rendered["image"]
        normal_pred = rendered["normal_map"]
        rgb_gt = batch["img"][0]
        normal_gt = batch["normal"][0]
        save_image(
            [rgb_gt, rgb_pred, normal_gt, normal_pred],
            os.path.join(self.render_path, f"render_{self.current_epoch:04d}.png"),
            nrow=2,
            padding=0,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        smpl_params = batch["smpl_params"]
        camera_params = batch["camera_params"]
        camera_param = {k: v[0] for k, v in camera_params.items()}

        gaussians = self.deformer(smpl_params)
        rendered = render(self.cfg, gaussians, camera_param)

        rgb_pred = rendered["image"].unsqueeze(0)  # (1,3, H, W)
        rgb_gt = batch["img"]

        normal_l1 = l1_loss(rendered["normal_map"], batch["normal"])
        normal_l2 = torch.mean((rendered["normal_map"] - batch["normal"]) ** 2)

        mse, psnr, ssim, lpips = self.evaluator.evaluate(rgb_pred, rgb_gt)
        metrics = {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "nc": mesh_normal_consistency(gaussians["posed_mesh"]).item(),
            "normal_l1": normal_l1.item(),
            "normal_l2": normal_l2.item(),
        }

        if batch_idx == 0:
            posed_mesh = gaussians["posed_mesh"]
            save_path = os.path.join(
                self.logger.log_dir, f"test/posed_mesh_{self.current_epoch}.obj"
            )
            IO().save_mesh(posed_mesh, save_path)

        if "scan_mesh" in batch:
            scan_mesh = batch["scan_mesh"]
            gt_mesh = Meshes(verts=scan_mesh["vertices"], faces=scan_mesh["faces"])
            cd, p2s = calculate_chamfer_p2s(gaussians["posed_mesh"], gt_mesh)
            metrics["cd"] = cd.item()
            metrics["p2s"] = p2s.item()

        self.test_step_outputs.append(
            {
                "batch": batch,
                "gaussians": gaussians,
                "rendered": rendered,
                "metrics": metrics,
            }
        )

    def on_test_end(self):

        metrics = [output["metrics"] for output in self.test_step_outputs]
        rendered = [output["rendered"] for output in self.test_step_outputs]
        batch = [output["batch"] for output in self.test_step_outputs]

        # 保存渲染视频
        frames = []
        for b, r in zip(batch, rendered):
            rgb_gt = b["img"][0]  # (3, H, W)
            normal_gt = b["normal"][0]  # (3, H, W)
            rgb_pred = r["image"]  # (3, H, W)
            normal_pred = r["normal_map"]  # (3, H, W)
            img_grid = make_grid(
                [rgb_gt, rgb_pred, normal_gt, normal_pred], nrow=2, padding=0
            )
            frame = (img_grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            frames.append(frame)
        video_path = os.path.join(self.test_dir, f"test_{self.current_epoch}.mp4")
        iio.imwrite(video_path, frames, fps=20)

        # 保存叠加视频
        overlay_frames = []
        alpha = 0.5  # rgb_pred的透明度
        for b, r in zip(batch, rendered):
            rgb_gt = b["img"][0]  # (3, H, W)
            rgb_pred = r["image"]  # (3, H, W)
            # 将rgb_pred以半透明方式叠加到rgb_gt上
            overlay = rgb_gt * (1 - alpha) + rgb_pred * alpha
            overlay = torch.clamp(overlay, 0, 1)
            frame = (overlay.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            overlay_frames.append(frame)
        noise = self.cfg.dataset.get("noise", 0.0)
        overlay_video_path = os.path.join(self.test_dir, f"test_overlay_{self.current_epoch}_{noise}.mp4")
        iio.imwrite(overlay_video_path, overlay_frames, fps=20)

        # 计算指标
        avg_metrics = {}
        for metric in metrics:
            for key, value in metric.items():
                if key not in avg_metrics:
                    avg_metrics[key] = 0.0
                avg_metrics[key] += value
        for key in avg_metrics:
            avg_metrics[key] /= len(self.test_step_outputs)

        # Print headers
        print(f"{'Key':<10} {'Value':<15}")
        print("-" * 25)
        # Print data
        keys = []
        values = []
        for key, value in avg_metrics.items():
            print(f"{key:<10} {str(value):<15}")
            keys.append(key)
            values.append(str(value))

        with open(os.path.join(self.test_dir, "metric.log"), "w") as f:
            f.write(", ".join(keys) + "\n" + ", ".join(values))

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        lrs = self.cfg.train

        params = [
            {"params": self.deformer.shs_dc, "lr": lrs.shs_lr},
            {"params": self.deformer.shs_rest, "lr": lrs.shs_lr / 20},
            {"params": self.deformer.opacity, "lr": lrs.opacity_lr},
            {
                "params": self.deformer.shape_encoder.parameters(),
                "lr": lrs.shape_encoder_lr,
            },
        ]
        if self.deformer.learnable_scale:
            params.append({"params": self.deformer._scales, "lr": lrs.scale_lr})
            params.append({"params": self.deformer._rotation, "lr": lrs.rotation_lr})

        # if self.deformer.use_vertex_gaussians:
        #     params.append({"params": self.deformer._scales_v, "lr": lrs.scale_lr})
        #     params.append({"params": self.deformer._rotation_v, "lr": lrs.rotation_lr})

        optimizer = torch.optim.Adam(params)
        return optimizer

    def compute_loss(self, batch, gaussians, rendered):
        """
        batch_data: dict_keys(['idx', 'img', 'mask', 'normal', 'bgcolor', 'smpl_params', 'camera_params'])
        pred: dict_keys(['image', 'depth', 'normal_map', 'scales', 'rotations', 'posed_mesh'])
        """
        gt_images = batch["img"]
        pred_images = rendered["image"]
        hp = self.cfg.train  # hyper parameters

        # gaussian loss
        lambda_ssim = hp.lambda_ssim
        l1_rgb = l1_loss(gt_images, pred_images)
        ssim_rgb = ssim(gt_images, pred_images)
        gs_loss = (1.0 - lambda_ssim) * l1_rgb + lambda_ssim * (1.0 - ssim_rgb)

        # mesh loss
        pred_mesh = gaussians["posed_mesh"]
        mesh_normal_consistency_loss = mesh_normal_consistency(pred_mesh)
        mesh_edges_loss = mesh_edge_loss(pred_mesh)
        mesh_loss = mesh_normal_consistency_loss + mesh_edges_loss

        # normal loss
        gt_normals = batch["normal"]
        pred_normals = rendered["normal_map"]
        l1_normal = l1_loss(gt_normals, pred_normals)
        ssim_normal = ssim(gt_normals, pred_normals)
        # mse_normal = torch.mean((gt_normals - pred_normals) ** 2)
        normal_loss = (1.0 - lambda_ssim) * l1_normal + lambda_ssim * (
            1.0 - ssim_normal
        )

        loss = {
            "l1_rgb": l1_rgb,
            "ssim_rgb": ssim_rgb,
            "gs_loss": gs_loss,
            "mesh_loss": mesh_loss,
            "normal_loss": normal_loss,
            "mesh_loss": mesh_loss,
        }

        total_loss = (
            hp.gs_loss_weight * gs_loss
            + hp.mesh_loss_weight * mesh_loss
            + hp.normal_loss_weight * normal_loss
        )

        # regularization loss
        if hp.reg_loss_weight > 0:
            v_offset = gaussians["v_offset"]
            reg_loss = torch.mean(v_offset**2)
            loss["reg_loss"] = reg_loss
            total_loss = total_loss + hp.reg_loss_weight * reg_loss

        # 三角等边约束
        if hp.edge_equal_loss_weight > 0:
            equal_edge_loss = compute_equilateral_loss_for_meshes(pred_mesh)
            loss["mesh_equal_edge"] = equal_edge_loss
            total_loss = total_loss + hp.edge_equal_loss_weight * equal_edge_loss

        # mesh_laplacian_loss
        if hp.mesh_laplacian_loss_weight > 0:
            mesh_laplacian_loss = mesh_laplacian_smoothing(pred_mesh)
            loss["mesh_laplacian_loss"] = mesh_laplacian_loss
            total_loss = (
                total_loss + hp.mesh_laplacian_loss_weight * mesh_laplacian_loss
            )

        loss["total_loss"] = total_loss

        return loss

    def predict_step(self, batch, batch_idx):
        smpl_params = batch["smpl_params"]
        camera_params = batch["camera_params"]
        camera_param = {k: v[0] for k, v in camera_params.items()}

        gaussians = self.deformer(smpl_params)
        # normalize the y-axis
        # max_y = gaussians["xyzs"][:, 1].max()
        # min_y = gaussians["xyzs"][:, 1].min()
        # gaussians["xyzs"][:, 1] = gaussians["xyzs"][:, 1] - (max_y + min_y) / 2

        rendered = render(self.cfg, gaussians, camera_param)

        output = (gaussians, rendered)
        return output

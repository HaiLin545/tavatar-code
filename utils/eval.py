import torch
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance


def point_mesh_distance(meshes, pcls):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = _PointFaceDistance.apply(
        points, points_first_idx, tris, tris_first_idx, max_points, 5e-3
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = torch.sqrt(point_to_face) * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist


def calculate_chamfer_p2s(src_mesh, tgt_mesh, num_samples=1000):

    tgt_points = Pointclouds(sample_points_from_meshes(src_mesh, num_samples))
    src_points = Pointclouds(sample_points_from_meshes(tgt_mesh, num_samples))
    p2s_dist = point_mesh_distance(src_mesh, tgt_points) * 100.0
    chamfer_dist = (point_mesh_distance(tgt_mesh, src_points) * 100.0 + p2s_dist) * 0.5

    return chamfer_dist, p2s_dist


class Evaluator(torch.nn.Module):
    """
    copied from https://github.com/zju3dv/neuralbody/blob/6bf1905822f71d1e568ef831110728fd1d06c94d/lib/evaluators/neural_volume.py
    adapted from https://github.com/escapefreeg/humannerf-eval/blob/master/eval.py
    """

    def __init__(self):
        super().__init__()
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").eval()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1).eval()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1).eval()

    @torch.no_grad()
    def evaluate(self, rgb_pred, rgb_gt):
        """
        rgb_pred: (1,3, H, W)
        rgb_gt: (1,3, H, W)
        """
        mse = torch.mean((rgb_pred - rgb_gt) ** 2).item()
        psnr = self.psnr_metric(rgb_pred, rgb_gt).item()
        ssim = self.ssim_metric(rgb_pred, rgb_gt).item() * 100
        lpips = self.lpips_metric(rgb_pred, rgb_gt).item()

        return mse, psnr, ssim, lpips

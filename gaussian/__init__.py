import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def rgb2shs(rgbs):
    C0 = 1 / (torch.sqrt(torch.tensor(4.0 * torch.pi)))
    return (rgbs - 0.5) / C0


def render(cfg, gaussians, camera_param, mode="train"):

    shs = gaussians["shs"]
    xyzs = gaussians["xyzs"]
    scales = gaussians["scales"]
    normals = gaussians["normals"]
    opacity = gaussians["opacity"]
    rotations = gaussians["quaternions"]
    
    means2D = torch.zeros_like(
        xyzs, dtype=xyzs.dtype, requires_grad=False, device=xyzs.device
    )
    raster_settings = GaussianRasterizationSettings(
        sh_degree=cfg.gaussian.max_sh_degree,
        prefiltered=False,
        debug=False,
        **camera_param,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    cov3D_precomp = None
    colors_precomp = None

    image, radii, depth, alpha = rasterizer(
        means3D=xyzs,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    normal_map = None
    if mode in ["train", "test"]:
        # convert normal to camera space
        cam_extrinsic = camera_param["viewmatrix"]
        R_w2c = cam_extrinsic[:3, :3]
        T_w2c = cam_extrinsic[:3, 3]
        normals = (R_w2c @ normals.T).T + T_w2c

        # get normal map
        normalized_f_normals = F.normalize(normals)
        normalized_f_normals = (normalized_f_normals + 1) / 2

        # match the coordinate system
        normalized_f_normals[:, 1] = 1 - normalized_f_normals[:, 1]
        normalized_f_normals[:, 2] = 1 - normalized_f_normals[:, 2]

        # rgb to shs
        normal_shs = rgb2shs(normalized_f_normals).reshape(-1, 1, 3)

        normal_raster_settings = GaussianRasterizationSettings(
            sh_degree=0, prefiltered=False, debug=False, **camera_param
        )
        normal_rasterizer = GaussianRasterizer(raster_settings=normal_raster_settings)
        normal_map, _, _, _ = normal_rasterizer(
            means3D=xyzs,
            means2D=means2D,
            shs=normal_shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    return {
        "image": image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": alpha,
        "normal_map": normal_map,
    }

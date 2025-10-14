import torch
import os
from pytorch3d.io import IO
import imageio.v3 as iio
from plyfile import PlyData, PlyElement
import numpy as np


def save_predict(outputs, save_dir, pose_name):
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"animate_{pose_name}.mp4")

    imgs = torch.stack([output[1]["image"] for output in outputs])
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    iio.imwrite(video_path, imgs, fps=25)

    gaussians = outputs[0][0]

    posed_meshes = gaussians["posed_mesh"]
    IO().save_mesh(
        posed_meshes,
        os.path.join(save_dir, f"pred_mesh_{pose_name}.obj"),
    )
    save_ply(
        gaussians=gaussians,
        path=os.path.join(save_dir, f"pred_gaussians_{pose_name}.ply"),
    )
    save_ply(
        gaussians=gaussians,
        path=os.path.join(save_dir, f"pred_gaussians_{pose_name}_rand.ply"),
        rand_color=True,
    )


@torch.no_grad()
def save_ply(gaussians, path, rand_color=False):

    xyz = gaussians["xyzs"].detach().cpu().numpy()
    normals = gaussians["normals"].detach().cpu().numpy()
    scale = torch.log(gaussians["scales"]).detach().cpu().numpy()
    rotation = gaussians["quaternions"].detach().cpu().numpy()
    shs = gaussians["shs"].detach()
    opacities = gaussians["opacity"].detach().cpu().numpy()

    if rand_color:
        shs = torch.rand_like(shs) * 2.0 - 1.0

    f_dc = shs[:, 0:1, :].flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = shs[:, 1:, :].flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [
        (attribute, "f4")
        for attribute in construct_list_of_attributes(shs, scale, rotation)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def construct_list_of_attributes(shs, scale, rotation):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(3):
        l.append("f_dc_{}".format(i))

    _, d, c = shs[:, 1:, :].shape
    for i in range(d * c):
        l.append("f_rest_{}".format(i))

    l.append("opacity")
    for i in range(scale.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(rotation.shape[1]):
        l.append("rot_{}".format(i))
    return l

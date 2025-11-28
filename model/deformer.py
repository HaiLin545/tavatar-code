import torch
import logging
from torch import nn
import torch.nn.functional as F
import submodules.smplx_modified as smplx
from submodules.smplx_modified.lbs import batch_rodrigues, batch_rigid_transform
from utils.smpl import gen_canonical_pose
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from gaussian import inverse_sigmoid
from pytorch3d.transforms import matrix_to_quaternion
from model.encoder import DisplacementEncoder
from utils.model import (
    get_normalized_vertex,
    get_init_complex_scale,
    compute_vertex_edge_length,
)

class HumanDeformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        canonical_pose = gen_canonical_pose(
            smpl_type=self.cfg.smpl.type, canonical_type=self.cfg.smpl.canonical_type
        )
        v_template, faces, J, A_inv, lbs_weights = self.init_smpl_model(
            canonical_pose, init_beta=None
        )

        # self.v_template = nn.Parameter(v_template.unsqueeze(0))  # (1, N, 3)

        self.register_buffer("v_template", v_template.unsqueeze(0))  # (1, N, 3)
        self.register_buffer("canonical_pose", canonical_pose)  # (1, 69)
        self.register_buffer("faces", faces.unsqueeze(0))  # [1, F, 3]
        self.register_buffer("A_inv", A_inv)
        self.register_buffer("J_canonical", J)  # (1, 45, 3)
        self.register_buffer("weights", lbs_weights.unsqueeze(0))  # (1, N, 24)

        self.surface_mesh_thickness = 1e-3 / (cfg.smpl.get("subdivide", 0) + 1)
        self.max_sh_degree = cfg.gaussian.get("max_sh_degree", 0)

        normalized_vertices = get_normalized_vertex(self.v_template)
        self.register_buffer("normalized_vertices", normalized_vertices)
        self.shape_encoder = DisplacementEncoder()

        self.learnable_scale = cfg.model.get("learnable_scale", False)
        self.use_vertex_gaussians = cfg.model.get("use_vertex_gaussians", False)
        self.min_edge_len_factor = cfg.model.get("min_edge_len_factor", 0.5)

        if self.learnable_scale:
            complex_numbers, scales = get_init_complex_scale(v_template, faces)
            self._rotation = nn.Parameter(complex_numbers)
            self._scales = nn.Parameter(torch.log(scales))

        # if self.use_vertex_gaussians:
        #     edge_lengths = compute_vertex_edge_length(v_template, faces, mode='avg')
        #     _rotation_v = torch.tensor([[1.0, 0.0]]).repeat(v_template.shape[0], 1)
        #     _scale_v = (
        #         edge_lengths.unsqueeze(-1).repeat(1, 2) * self.min_edge_len_factor
        #     )
        #     self._rotation_v = nn.Parameter(_rotation_v)
        #     self._scales_v = nn.Parameter(torch.log(_scale_v))

        opacity = torch.ones((self.n_gs, 1), dtype=torch.float) * 0.9999
        # self.register_buffer("opacity", inverse_sigmoid(opacity))
        self.opacity = nn.Parameter(inverse_sigmoid(opacity))
        self.shs_dc = nn.Parameter(torch.zeros([self.n_gs, 1, 3]))
        self.shs_rest = nn.Parameter(
            torch.zeros([self.n_gs, (self.max_sh_degree + 1) ** 2 - 1, 3])
        )

    @torch.no_grad()
    def init_smpl_model(
        self,
        canonical_pose,
        init_beta=None,
    ):

        self.body_model = smplx.create(
            model_path=self.cfg.smpl.path,
            model_type=self.cfg.smpl.type,
            gender=self.cfg.smpl.gender,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.body_model.requires_grad_(False)

        if init_beta is None:
            init_beta = torch.zeros((1, self.body_model.num_betas), dtype=torch.float32)

        smpl_output = self.body_model(
            betas=init_beta, body_pose=canonical_pose, return_extra=True
        )

        v_template = smpl_output.vertices
        v_center = v_template[0].mean(dim=0)
        v_template = v_template - v_center
        J = smpl_output.extra["J"] - v_center
        faces = torch.from_numpy(self.body_model.faces.astype(int))
        A_inv = torch.linalg.inv(smpl_output.extra["A"])
        lbs_weights = self.body_model.lbs_weights

        pt3d_mesh = Meshes(verts=v_template, faces=faces.unsqueeze(0))
        pt3d_subdivide_mesh = SubdivideMeshes()

        if self.cfg.smpl.subdivide > 0:
            logging.info(f"Subdividing the mesh {self.cfg.smpl.subdivide } times")
            for _ in range(self.cfg.smpl.subdivide):
                pt3d_mesh, lbs_weights = pt3d_subdivide_mesh(
                    pt3d_mesh, feats=lbs_weights
                )

        v_template = pt3d_mesh.verts_packed()
        faces = pt3d_mesh.faces_packed()
        logging.info(
            f"Template mesh has {v_template.shape[0]} vertices and {faces.shape[0]} faces"
        )

        return v_template, faces, J, A_inv, lbs_weights

    def forward(self, smpl_param, **kwargs):

        full_pose = smpl_param["full_pose"]  # [1, 72]
        transl = smpl_param["transl"].unsqueeze(0)

        if self.cfg.smpl.type == "smplx":
            full_pose = full_pose + self.body_model.pose_mean

        R, t = self.lbs(full_pose, weights=self.weights)

        # verts = self.get_vertex()

        v_offset = self.shape_encoder(self.normalized_vertices)
        verts = self.v_template + v_offset

        verts_posed = torch.einsum("bnij,bnj->bni", R, verts) + t + transl

        posed_mesh = Meshes(verts=verts_posed, faces=self.faces)
        xyzs, scales, normals, quaternions = self.get_face_gaussians(
            verts_posed, posed_mesh
        )

        if self.use_vertex_gaussians:
            (
                xyzs_v,
                scales_v,
                normals_v,
                quaternions_v,
            ) = self.get_vertex_gaussians(verts_posed, posed_mesh)

            xyzs = torch.cat([xyzs, xyzs_v], dim=0)
            scales = torch.cat([scales, scales_v], dim=0)
            normals = torch.cat([normals, normals_v], dim=0)
            quaternions = torch.cat([quaternions, quaternions_v], dim=0)

        output = {
            "xyzs": xyzs,
            "scales": scales,
            "normals": normals,
            "shs": self.get_shs(),
            "posed_mesh": posed_mesh,
            "quaternions": quaternions,
            "opacity": torch.sigmoid(self.opacity),
            "v_offset": v_offset,
        }

        return output

    def lbs(self, pose, weights):
        B = pose.shape[0]
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([B, -1, 3, 3])
        J_transformed, rel_A = batch_rigid_transform(
            rot_mats,
            self.J_canonical.repeat([B, 1, 1]),
            self.body_model.parents,
        )
        W = weights.repeat(B, 1, 1)
        A = torch.einsum("bnij,bnjk->bnik", rel_A, self.A_inv.expand([B, -1, 4, 4]))
        T = torch.einsum("bnj, bjrc -> bnrc", W, A)

        R, t = T[:, :, :3, :3], T[:, :, :3, 3]
        return R, t

    def get_vertex(self):

        v_offset = self.shape_encoder(self.normalized_vertices)
        v = self.v_template + v_offset
        # v = self.v_template
        return v

    @property
    def n_gs(self):

        if self.use_vertex_gaussians:
            return self.v_template.shape[1] + self.faces.shape[1]
        else:
            return self.faces.shape[1]

    def get_shs(self):
        shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        return shs

    def get_xyzs(self, verts_posed):
        """get the center of each face as the xyz of each gaussian"""
        faces_verts = verts_posed[:, self.faces[0]]
        xyzs = faces_verts.mean(dim=2)[0]  # (F, 3)
        return xyzs

    def get_face_gaussians(self, verts_posed, posed_mesh):

        if self.learnable_scale:
            xyzs = self.get_xyzs(verts_posed)
            plane_scales = torch.exp(self._scales)
            normal_scale = (
                torch.ones(len(self._scales), 1, device=xyzs.device)
                * self.surface_mesh_thickness
            )
            scales = torch.cat([normal_scale, plane_scales], dim=-1)
            normals = self.get_face_normal(verts_posed[0], self.faces[0])
            quaternions = self.get_quaternions(verts_posed[0], self.faces[0], normals)

        else:
            xyzs, scales = self.get_inner_circle(verts_posed[0], self.faces[0])
            normals, quaternions = self.get_incircle_quaterinions_normal(
                verts_posed[0], self.faces[0]
            )

        return xyzs, scales, normals, quaternions

    def get_vertex_gaussians(self, verts_posed, posed_mesh):

        xyzs_v = verts_posed[0]

        edge_len = compute_vertex_edge_length(verts_posed[0], self.faces[0], mode='avg')
        # plane_scales = torch.exp(self._scales_v)
        plane_scales = (
            edge_len.unsqueeze(-1).repeat(1, 2) * self.min_edge_len_factor
        )
        normal_scale = (
            torch.ones(len(plane_scales), 1, device=xyzs_v.device)
            * self.surface_mesh_thickness
        )
        scales_v = torch.cat([plane_scales, normal_scale], dim=-1)
        normals_v = posed_mesh.verts_normals_packed()
        quaternions_v = self.get_quaternions_v(normals_v)

        return xyzs_v, scales_v, normals_v, quaternions_v

    def get_inner_circle(self, vertices, faces):
        v0 = vertices[faces[:, 0], :]  # F, 3
        v1 = vertices[faces[:, 1], :]
        v2 = vertices[faces[:, 2], :]
        edge1 = v1 - v0  # 从v0到v1的向量
        edge2 = v2 - v0  # 从v0到v2的向量
        edge3 = v2 - v1  # 从v1到v2的向量
        cross_product = torch.cross(edge1, edge2, dim=-1)
        area = 0.5 * torch.norm(cross_product, dim=-1)

        # 计算边长
        eps = 1e-10
        a = torch.clamp(torch.norm(edge3, dim=-1), min=eps)  # |v2 - v1|
        b = torch.clamp(torch.norm(edge2, dim=-1), min=eps)  # |v2 - v0|
        c = torch.clamp(torch.norm(edge1, dim=-1), min=eps)  # |v1 - v0|

        # 计算半周长
        s = (a + b + c) / 2
        s = torch.clamp(s, min=eps)
        area = torch.clamp(area, min=eps)  # 确保面积为正
        inradius = area / s

        # 计算内心坐标 - 使用边长加权
        a_unsqueezed = a.unsqueeze(-1)
        b_unsqueezed = b.unsqueeze(-1)
        c_unsqueezed = c.unsqueeze(-1)
        perimeter = torch.clamp(a_unsqueezed + b_unsqueezed + c_unsqueezed, min=eps)
        incenter = (
            a_unsqueezed * v0 + b_unsqueezed * v1 + c_unsqueezed * v2
        ) / perimeter

        plane_scales = inradius.unsqueeze(-1).repeat(1, 2)
        normal_scale = self.surface_mesh_thickness * torch.ones(
            inradius.shape[0], 1, device=vertices.device
        )
        scales = torch.cat([plane_scales, normal_scale], dim=-1)

        return incenter, scales

    def get_incircle_quaterinions_normal(self, vertices, faces):
        v0 = vertices[faces[:, 0], :]
        v1 = vertices[faces[:, 1], :]
        v2 = vertices[faces[:, 2], :]
        # 计算边向量
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = torch.linalg.cross(edge1, edge2)

        z = normal / (normal.norm(dim=-1, keepdim=True))
        x = edge1 / (edge1.norm(dim=-1, keepdim=True))
        y = torch.linalg.cross(z, x)
        y = y / y.norm(dim=-1, keepdim=True)

        R = torch.cat((x, y, z), dim=-1)
        R = R.view(-1, 3, 3).transpose(1, 2)  # B, 3, 3
        quaterinions = matrix_to_quaternion(R)

        return normal, quaterinions

    def get_quaternions(self, verts, faces, face_normals):
        """
        verts: (N, 3)
        faces: (F, 3)
        """

        R_0 = torch.nn.functional.normalize(face_normals, dim=-1)
        # We use the first side of every triangle as the second base axis
        faces_verts = verts[faces]
        base_R_1 = torch.nn.functional.normalize(
            faces_verts[:, 0] - faces_verts[:, 1], dim=-1
        )
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
        # We now apply the learned 2D rotation to the base quaternion
        complex_numbers = torch.nn.functional.normalize(self._rotation, dim=-1)
        t1 = complex_numbers[:, 0:1]
        t2 = complex_numbers[:, 1:2]
        R_1 = t1 * base_R_1 + t2 * base_R_2
        R_2 = -t2 * base_R_1 + t1 * base_R_2

        # We concatenate the three vectors to get the rotation matrix
        R = torch.cat([R_0, R_1, R_2], dim=-1)
        R = R.view(-1, 3, 3).transpose(1, 2)  # B, 3, 3

        quaternion = matrix_to_quaternion(R)
        return quaternion

    def get_face_normal(self, vertices, faces):
        v0 = vertices[faces[:, 0], :]
        v1 = vertices[faces[:, 1], :]
        v2 = vertices[faces[:, 2], :]
        # 计算边向量
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = torch.cross(edge1, edge2)
        normal = normal / (normal.norm(dim=-1, keepdim=True))
        return normal

    def get_quaternions_v(self, normals):
        """
        计算四元数，使得Z轴与给定的法线对齐。
        """
        # z 轴现在是我们的法线方向
        z = F.normalize(normals, dim=-1)

        # --- 构建正交基，以 z 轴为基准 ---
        # 找到绝对值最小的分量
        min_abs_idx = torch.argmin(torch.abs(z), dim=-1)
        # 创建一个与 z 最不平行的向量 c
        c = torch.zeros_like(z)
        c.scatter_(1, min_abs_idx.unsqueeze(1), 1.0)

        # 使用叉乘构建正交的 x 和 y 轴
        x = F.normalize(torch.cross(c, z, dim=-1), dim=-1)
        y = F.normalize(torch.cross(z, x, dim=-1), dim=-1)
        # complex_numbers = F.normalize(self._rotation_v, dim=-1)
        # t1 = complex_numbers[:, 0:1]  # cos(theta)
        # t2 = complex_numbers[:, 1:2]  # sin(theta)
        # R_1 = t1 * x + t2 * y
        # R_2 = -t2 * x + t1 * y
        R = torch.stack([x, y, z], dim=2)  # 形状 (B, 3, 3)
        quaternion = matrix_to_quaternion(R)

        return quaternion

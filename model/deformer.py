import torch
import logging
from torch import nn
import submodules.smplx_modified as smplx
from submodules.smplx_modified.lbs import batch_rodrigues, batch_rigid_transform
from utils.smpl import gen_canonical_pose
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes
from gaussian import inverse_sigmoid
from pytorch3d.transforms import matrix_to_quaternion
from model.encoder import DisplacementEncoder


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

        self.register_buffer('v_template', v_template.unsqueeze(0))  # (1, N, 3)
        self.register_buffer("canonical_pose", canonical_pose)  # (1, 69)
        self.register_buffer("faces", faces.unsqueeze(0))  # [1, F, 3]
        self.register_buffer("A_inv", A_inv)
        self.register_buffer("J_canonical", J)  # (1, 45, 3)
        self.register_buffer("weights", lbs_weights.unsqueeze(0))  # (1, N, 24)

        opacity = torch.ones((self.n_gs, 1), dtype=torch.float) * 0.9999
        self.register_buffer("opacity", inverse_sigmoid(opacity))

        self.surface_mesh_thickness = 1e-3 / (cfg.smpl.get("subdivide", 0) + 1)
        self.max_sh_degree = cfg.gaussian.get("max_sh_degree", 0)
        self.shs_dc = nn.Parameter(torch.zeros([self.n_gs, 1, 3]))
        self.shs_rest = nn.Parameter(
            torch.zeros([self.n_gs, (self.max_sh_degree + 1) ** 2 - 1, 3])
        )

        v_template_centered = self.v_template - self.v_template.mean(
            dim=1, keepdim=True
        )
        minmax = [
            v_template_centered[0].min(dim=0).values * 1.05,
            v_template_centered[0].max(dim=0).values * 1.05,
        ]
        normalized_vertices = (v_template_centered - minmax[0]) / (
            minmax[1] - minmax[0]
        )
        self.register_buffer("normalized_vertices", normalized_vertices)
        self.shape_encoder = DisplacementEncoder()

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
        verts = self.get_vertex()
        verts_posed = torch.einsum("bnij,bnj->bni", R, verts) + t + transl
        xyzs, scales = self.get_inner_circle(verts_posed[0], self.faces[0])
        normals, quaternions = self.get_quaterinions_normal(
            verts_posed[0], self.faces[0]
        )

        posed_mesh = Meshes(verts=verts_posed, faces=self.faces)
        output = {
            "xyzs": xyzs,
            "scales": scales,
            "normals": normals,
            "shs": self.get_shs(),
            "posed_mesh": posed_mesh,
            "quaternions": quaternions,
            "opacity": torch.sigmoid(self.opacity),
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
        return self.faces.shape[1]

    def get_shs(self):
        shs = torch.cat([self.shs_dc, self.shs_rest], dim=1)
        return shs

    def get_xyzs(self, verts_posed):
        """get the center of each face as the xyz of each gaussian"""
        faces_verts = verts_posed[:, self.faces[0]]
        xyzs = faces_verts.mean(dim=2)
        return xyzs

    def get_gaussians(self, verts_posed):

        posed_mesh = Meshes(verts=verts_posed, faces=self.faces)

        xyzs, scales = self.get_inner_circle(verts_posed[0], self.faces[0])
        normals, quaternions = self.get_quaterinions_normal(
            verts_posed[0], self.faces[0]
        )

        # if self.use_point_gs:

        #     scales_v = self.get_scales_v
        #     normals_v = posed_mesh.verts_normals_packed()

        #     # quaternions_v = self._rotations_v
        #     quaternions_v = self.get_quaternions_v(posed_mesh.verts_normals_packed())

        #     xyzs = torch.cat([xyzs, verts_posed[0]])
        #     scales = torch.cat([scales, scales_v])
        #     quaternions = torch.cat([quaternions, quaternions_v])
        #     normals = torch.cat([normals, normals_v])

        # shs = self.get_shs_nomal(normals)
        shs = self.get_shs

        return xyzs, scales, shs, normals, quaternions, posed_mesh

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

    def get_quaterinions_normal(self, vertices, faces):
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

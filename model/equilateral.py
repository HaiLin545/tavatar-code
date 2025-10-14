import torch
from pytorch3d.structures import Meshes


def compute_angle_loss(vec_a, vec_b):
    cos_theta = torch.sum(vec_a * vec_b, dim=1) / (
        torch.norm(vec_a, dim=1) * torch.norm(vec_b, dim=1) + 1e-8
    )
    return (1 - cos_theta) ** 2  # 平方误差鼓励cosθ接近0.5


def compute_equilateral_loss_for_meshes(meshes: Meshes) -> torch.Tensor:
    """
    计算Meshes对象中等边三角形损失（基于边长一致性和角度约束）
    输入: pytorch3d.Meshes对象
    输出: 总损失值（标量）
    """
    # 提取有效顶点和面（自动处理batch和padding）
    verts_packed = meshes.verts_packed()  # [V, 3]
    faces_packed = meshes.faces_packed()  # [F, 3]

    if faces_packed.shape[0] == 0:
        return torch.tensor(0.0, device=meshes.device)

    # 获取每个面的顶点坐标 [F, 3, 3]
    faces_verts = verts_packed[faces_packed]  # (num_faces, 3, 3)

    # 分解三个顶点的坐标
    p0 = faces_verts[:, 0, :]  # (F, 3)
    p1 = faces_verts[:, 1, :]  # (F, 3)
    p2 = faces_verts[:, 2, :]  # (F, 3)

    # ----------- 计算边长方差项 -----------
    # 计算三条边的向量及长度
    e0 = p1 - p0  # 边 p0 -> p1
    e1 = p2 - p1  # 边 p1 -> p2
    e2 = p0 - p2  # 边 p2 -> p0

    l0 = torch.norm(e0, dim=1)  # (F,)
    l1 = torch.norm(e1, dim=1)  # (F,)
    l2 = torch.norm(e2, dim=1)  # (F,)

    # 计算每个面的边长方差（方差越小，边长越一致）
    lengths = torch.stack([l0, l1, l2], dim=1)  # (F, 3)
    var_per_face = torch.var(lengths, dim=1, unbiased=False)  # (F,)

    # 角0（顶点p0处的角）：向量由p0→p1和p0→p2
    vec0_a = e0  # p0→p1
    vec0_b = p2 - p0  # p0→p2
    term0 = compute_angle_loss(vec0_a, vec0_b)

    # 角1（顶点p1处的角）：向量由p1→p2和p1→p0
    vec1_a = e1  # p1→p2
    vec1_b = -e0  # p1←p0
    term1 = compute_angle_loss(vec1_a, vec1_b)

    # 角2（顶点p2处的角）：向量由p2→p0和p2→p1
    vec2_a = e2  # p2→p0
    vec2_b = -e1  # p2←p1
    term2 = compute_angle_loss(vec2_a, vec2_b)

    angle_term_per_face = term0 + term1 + term2  # (F,)

    # 总损失：边长方差项 + 角度项，并对所有面求和
    total_loss = (var_per_face + angle_term_per_face).mean()
    return total_loss

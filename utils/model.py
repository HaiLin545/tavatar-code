import torch
from torch_scatter import scatter_min, scatter_max


def get_init_complex_scale(vertices, faces):
    n_points = faces.shape[0]
    faces_verts = vertices[faces]
    edges = faces_verts - faces_verts[:, [1, 2, 0]]
    min_edge = edges.norm(dim=-1).min(dim=-1)[0]
    scales = min_edge * 0.5
    scales = scales.unsqueeze(-1).repeat(1, 2)

    complex_numbers = torch.zeros(n_points, 2).to(vertices.device)
    complex_numbers[:, 0] = 1.0

    return complex_numbers, scales


def get_normalized_vertex(vertices):
    centered_vertices = vertices - vertices.mean(dim=1, keepdim=True)
    min = centered_vertices[0].min(dim=0).values
    max = centered_vertices[0].max(dim=0).values
    normalized_vertices = (centered_vertices - min) / (max - min)
    return normalized_vertices


def compute_vertex_edge_length(vertices, faces, mode='min'):
    """
    计算与每个顶点相连的所有边中的最短边长。

    参数:
        vertices (torch.Tensor): 顶点坐标张量，形状为 (num_vertices, 3)。
        faces (torch.Tensor): 面索引张量，形状为 (num_faces, 3)。

    返回:
        torch.Tensor: 一个长度为 num_vertices 的张量，其中每个元素是
                      对应顶点的最短连接边长。
    """
    num_vertices = vertices.shape[0]

    # 1. 从面索引中提取所有的边
    # 每个面 (v1, v2, v3) 包含三条边: (v1, v2), (v2, v3), (v3, v1)
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)

    # 2. 计算所有边的长度
    # 使用 advanced indexing 获取每条边端点的坐标
    v_a = vertices[edges[:, 0]]
    v_b = vertices[edges[:, 1]]
    edge_lengths = torch.norm(v_a - v_b, dim=-1)

    # 3. 为聚合做准备
    all_vertex_indices = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    repeated_edge_lengths = torch.cat([edge_lengths, edge_lengths], dim=0)

    if mode == 'min':
        min_lengths = scatter_min(
            repeated_edge_lengths, all_vertex_indices, dim_size=num_vertices
        )[0]
        return min_lengths
    
    if mode == 'max':
        max_lengths = scatter_max(
            repeated_edge_lengths, all_vertex_indices, dim_size=num_vertices
        )[0]
        return max_lengths

    if mode == 'avg':
        sum_lengths = torch.zeros(num_vertices, device=vertices.device)
        count_lengths = torch.zeros(num_vertices, device=vertices.device)

        sum_lengths = sum_lengths.index_add(0, all_vertex_indices, repeated_edge_lengths)
        count_lengths = count_lengths.index_add(0, all_vertex_indices, torch.ones_like(repeated_edge_lengths))

        average_lengths = sum_lengths / count_lengths.clamp(min=1)
        return average_lengths


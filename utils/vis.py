import torch
import numpy as np
import colorsys

def weights_to_rgb(weights):
    """
    weights: (N, C) tensor
    Maps each channel C to a unique color and blends them.
    Returns: (N, 3) RGB values in [0, 1]
    """
    N, C = weights.shape
    device = weights.device
    
    # Generate C distinct colors using HSV space
    # This ensures consistent coloring for each joint index
    palette = []
    for i in range(C):
        # Hue distributed evenly
        hue = i / C
        # High saturation and value for vivid colors
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        palette.append([r, g, b])
    
    palette = torch.tensor(palette, device=device, dtype=torch.float32) # (C, 3)
    
    # Weighted sum of colors: (N, C) @ (C, 3) -> (N, 3)
    rgb = weights @ palette
    
    # Clamp to ensure valid range
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb

def weights_to_pca_rgb(weights):
    """
    weights: (N, C) tensor
    Returns: (N, 3) RGB values in [0, 1]
    """
    # PCA
    # Center the data
    mean = weights.mean(dim=0, keepdim=True)
    centered = weights - mean

    # SVD on covariance (C, C) is fast
    # U, S, V = torch.svd(centered.T @ centered)
    # U is (C, C). Top 3 components are U[:, :3]
    U, S, V = torch.svd(centered.T @ centered)
    pca_weights = centered @ U[:, :3]  # Project (N, 3)

    # Normalize to [0, 1]
    p_min = pca_weights.min(dim=0, keepdim=True)[0]
    p_max = pca_weights.max(dim=0, keepdim=True)[0]
    pca_rgb = (pca_weights - p_min) / (p_max - p_min + 1e-8)
    
    return pca_rgb

def save_ply_with_color(filename, points, colors):
    """
    points: (N, 3) numpy array
    colors: (N, 3) numpy array, uint8 or float [0,1]
    """
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if colors.max() <= 1.0 + 1e-6:
        colors = (colors * 255).astype('uint8')
    else:
        colors = colors.astype('uint8')
        
    data = np.hstack([points, colors])
    
    header = f"""ply
format ascii 1.0
element vertex {points.shape[0]}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")
    print(f"Saved {filename}")

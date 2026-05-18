import torch
import numpy as np


def create_voxel_grid_torch(res_x, res_y, res_z, bounds, dtype=torch.float32, device='cpu'):
    """
    Create a voxel grid with vertices and cube indices (vectorized).
    
    Args:
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        bounds: Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        dtype: Data type for vertex coordinates (default: torch.float32)
    
    Returns:
        grids: Tensor of shape (num_vertices, 3) containing vertex coordinates
        cubes: Tensor of shape (num_cubes, 8) containing vertex indices for each cube
    """
    
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    
    # Create 1D coordinates for each dimension
    x_coords = torch.linspace(x_min, x_max, res_x + 1, dtype=dtype, device=device)
    y_coords = torch.linspace(y_min, y_max, res_y + 1, dtype=dtype, device=device)
    z_coords = torch.linspace(z_min, z_max, res_z + 1, dtype=dtype, device=device)
    
    # Create meshgrid: shape (res_z+1, res_y+1, res_x+1)
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Flatten to get all vertices: shape (num_vertices, 3)
    grids = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    # Vectorized cube indices generation
    stride_x = 1
    stride_y = res_x + 1
    stride_z = (res_y + 1) * (res_x + 1)
    
    # Create all cube positions via meshgrid
    i_idx = torch.arange(res_x, device=device)
    j_idx = torch.arange(res_y, device=device)
    k_idx = torch.arange(res_z, device=device)

    k_grid, j_grid, i_grid = torch.meshgrid(k_idx, j_idx, i_idx, indexing='ij')
    
    # Compute base index for each cube
    base = k_grid * stride_z + j_grid * stride_y + i_grid * stride_x
    base = base.flatten().unsqueeze(1)  # shape (num_cubes, 1)
    
    # Define vertex offsets within a cube
    offsets = torch.tensor([
        [0, 0, 0],  # v0
        [1, 0, 0],  # v1
        [0, 1, 0],  # v2
        [1, 1, 0],  # v3
        [0, 0, 1],  # v4
        [1, 0, 1],  # v5
        [0, 1, 1],  # v6
        [1, 1, 1],  # v7
    ], dtype=torch.int32, device=device)
    
    # Convert offsets to linear indices
    vertex_offsets = offsets[:, 0] * stride_x + offsets[:, 1] * stride_y + offsets[:, 2] * stride_z
    
    # Broadcast and add: shape (num_cubes, 8)
    cubes = base + vertex_offsets.unsqueeze(0)
    cubes = cubes.int()
    
    return grids, cubes

def create_voxel_grid_numpy(res_x, res_y, res_z, bounds, dtype=np.float32):
    """
    Create a voxel grid with vertices and cube indices (vectorized).
    
    Args:
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        bounds: Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        dtype: Data type for vertex coordinates (default: np.float32)
    
    Returns:
        grids: Numpy array of shape (num_vertices, 3) containing vertex coordinates
        cubes: Numpy array of shape (num_cubes, 8) containing vertex indices for each cube
    """
    
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    
    # Create 1D coordinates for each dimension
    x_coords = np.linspace(x_min, x_max, res_x + 1, dtype=dtype)
    y_coords = np.linspace(y_min, y_max, res_y + 1, dtype=dtype)
    z_coords = np.linspace(z_min, z_max, res_z + 1, dtype=dtype)
    
    # Create meshgrid: shape (res_z+1, res_y+1, res_x+1)
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Flatten to get all vertices: shape (num_vertices, 3)
    grids = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    # Vectorized cube indices generation
    stride_x = 1
    stride_y = res_x + 1
    stride_z = (res_y + 1) * (res_x + 1)
    
    # Create all cube positions via meshgrid
    i_idx = np.arange(res_x)
    j_idx = np.arange(res_y)
    k_idx = np.arange(res_z)
    
    k_grid, j_grid, i_grid = np.meshgrid(k_idx, j_idx, i_idx, indexing='ij')
    
    # Compute base index for each cube
    base = k_grid * stride_z + j_grid * stride_y + i_grid * stride_x
    base = np.expand_dims(base.flatten(), axis=1)  # shape (num_cubes, 1)
    
    # Define vertex offsets within a cube
    offsets = np.array([
        [0, 0, 0],  # v0
        [1, 0, 0],  # v1
        [0, 1, 0],  # v2
        [1, 1, 0],  # v3
        [0, 0, 1],  # v4
        [1, 0, 1],  # v5
        [0, 1, 1],  # v6
        [1, 1, 1],  # v7
    ], dtype=np.int32)
    
    # Convert offsets to linear indices
    vertex_offsets = offsets[:, 0] * stride_x + offsets[:, 1] * stride_y + offsets[:, 2] * stride_z
    
    # Broadcast and add: shape (num_cubes, 8)
    cubes = base + np.expand_dims(vertex_offsets, axis=0)
    cubes = cubes.astype(np.int32)
    
    return grids, cubes

def create_voxel_grid(res_x, res_y, res_z, bounds, dtype=np.float32, device='cpu', return_numpy=True):
    if return_numpy:
        return create_voxel_grid_numpy(res_x, res_y, res_z, bounds, dtype=dtype)
    else:
        return create_voxel_grid_torch(res_x, res_y, res_z, bounds, dtype=dtype, device=device)
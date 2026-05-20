import torch
import numpy as np

from .._C import pc_to_voxel_grid as pc_to_voxel_grid_cuda


def create_voxel_grid_torch(res_x, res_y, res_z, bounds, dtype=torch.float32, device='cpu'):
    """
    Create a voxel grid with vertices and cube indices (vectorized).
    
    Args:
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        bounds: Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        dtype: Data type for vertex coordinates (default: torch.float32)
        device: Device for tensor operations (default: 'cpu')
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
    
def drop_grid_w_mesh(grid_vertices, grid_cubes, mesh_vertices, mesh_faces, chunk_size=10000, margin_ratio=0.0):
    """
    Drop grid vertices and cubes, that are not intersecting with the mesh.
    
    Args:
        grid_vertices: Tensor of shape (num_vertices, 3) containing vertex coordinates
        grid_cubes: Tensor of shape (num_cubes, 8) containing vertex indices for each cube
        mesh_vertices: Tensor of shape (num_mesh_vertices, 3) containing mesh vertex coordinates
        mesh_faces: Tensor of shape (num_mesh_faces, 3) containing indices of vertices for each face
        chunk_size: Size of chunks for processing to prevent OOM (default: 10000)
        margin_ratio: Ratio to expand bounding boxes. margin = margin_ratio * (max - min) (default: 0.0, no expansion)
    Returns:
        drop_grid_vertices: Tensor of shape (num_dropped_vertices, 3) containing dropped vertex coordinates
        drop_grid_cubes: Tensor of shape (num_dropped_cubes, 8) containing vertex indices for each dropped cube
    """
    
    device = grid_vertices.device
    
    # 1. Find bounding boxes for each cube
    # Using v0 for min and v7 for max based on standard offset creation logic in `create_voxel_grid`
    cube_min = grid_vertices[grid_cubes[:, 0]]
    cube_max = grid_vertices[grid_cubes[:, 7]]
    
    # Apply margin to cube bounding boxes
    if margin_ratio > 0.0:
        cube_margin = margin_ratio * (cube_max - cube_min)
        cube_min = cube_min - cube_margin
        cube_max = cube_max + cube_margin

    # 2. Find bounding boxes for each mesh triangle
    triangles = mesh_vertices[mesh_faces]  # (num_faces, 3, 3)
    tri_min = triangles.min(dim=1)[0]      # (num_faces, 3)
    tri_max = triangles.max(dim=1)[0]      # (num_faces, 3)
    
    # Apply margin to triangle bounding boxes
    if margin_ratio > 0.0:
        tri_margin = margin_ratio * (tri_max - tri_min)
        tri_min = tri_min - tri_margin
        tri_max = tri_max + tri_margin

    num_cubes = grid_cubes.shape[0]
    num_faces = mesh_faces.shape[0]
    active_cubes_mask = torch.zeros(num_cubes, dtype=torch.bool, device=device)

    # 3. Check for AABB Bounding Box collisions in chunks to prevent OOM
    for i in range(0, num_faces, chunk_size):
        chunk_tri_min = tri_min[i : i + chunk_size].unsqueeze(0)  # (1, chunk, 3)
        chunk_tri_max = tri_max[i : i + chunk_size].unsqueeze(0)  # (1, chunk, 3)
        
        # Bounding boxes overlap if (A.min <= B.max) and (A.max >= B.min) across all 3 dimensions
        overlap = (cube_min.unsqueeze(1) < chunk_tri_max) & \
                  (cube_max.unsqueeze(1) > chunk_tri_min)
        
        overlap = overlap.all(dim=-1)       # True if x, y, and z all overlap
        
        # If the cube overlaps with ANY face in this chunk, mark it as active
        active_cubes_mask |= overlap.any(dim=1)

    # Filter out inactive cubes
    drop_grid_cubes = grid_cubes[active_cubes_mask]
    
    # 4. Remove unused vertices and remap the indices
    unique_vertex_indices, inverse_indices = torch.unique(drop_grid_cubes, return_inverse=True)
    
    inverse_indices = inverse_indices.to(drop_grid_cubes.dtype)
    
    drop_grid_vertices = grid_vertices[unique_vertex_indices]
    drop_grid_cubes = inverse_indices.reshape(drop_grid_cubes.shape)
        
    return drop_grid_vertices, drop_grid_cubes, unique_vertex_indices

def pc_to_voxel_grid(points, res_x, res_y, res_z, k_threshold=1, num_keep=0):
    """
    Convert a point cloud to a sparse voxel grid.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        k_threshold: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
    
    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    return pc_to_voxel_grid_cuda(points, res_x, res_y, res_z, k_threshold, num_keep)
import torch
import numpy as np

from .._C import pc_to_voxel_grid as pc_to_voxel_grid_cuda
from .._C import pc_to_voxel_grid_long as pc_to_voxel_grid_long_cuda
from .._C import pc_to_voxel_grid_chunk as pc_to_voxel_grid_chunk_cuda
from .._C import pc_to_voxel_grid_chunk_long as pc_to_voxel_grid_chunk_long_cuda


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

def pc_to_voxel_grid(points, res_x, res_y, res_z, k=1, num_keep=0, rmi_x=1.0, rmi_y=1.0, rmi_z=1.0, rma_x=1.0, rma_y=1.0, rma_z=1.0):
    """
    Convert a point cloud to a sparse voxel grid.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        k: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
        rmi_x, rmi_y, rmi_z: Minimum keeping ratio of the minimum bounding box
        rma_x, rma_y, rma_z: Maximum keeping ratio of the maximum bounding box

    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    return pc_to_voxel_grid_cuda(points, res_x, res_y, res_z, k, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z)

def pc_to_voxel_grid_long(points, res_x, res_y, res_z, k=1, num_keep=0, rmi_x=1.0, rmi_y=1.0, rmi_z=1.0, rma_x=1.0, rma_y=1.0, rma_z=1.0):
    """
    Convert a point cloud to a sparse voxel grid, using 64-bit indices.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        k: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
        rmi_x, rmi_y, rmi_z: Minimum keeping ratio of the minimum bounding box
        rma_x, rma_y, rma_z: Maximum keeping ratio of the maximum bounding box

    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    return pc_to_voxel_grid_long_cuda(points, res_x, res_y, res_z, k, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z)

def pc_to_voxel_grid_chunk(points, res_x, res_y, res_z, chunk_size=256, k=1, num_keep=0, rmi_x=1.0, rmi_y=1.0, rmi_z=1.0, rma_x=1.0, rma_y=1.0, rma_z=1.0):
    """
    Convert a point cloud to a sparse voxel grid using spatial chunking.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        chunk_size: Size of the cubic chunk to process at one time to save VRAM (default: 256)
        k: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
        rmi_x, rmi_y, rmi_z: Minimum keeping ratio of the minimum bounding box
        rma_x, rma_y, rma_z: Maximum keeping ratio of the maximum bounding box

    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    return pc_to_voxel_grid_chunk_cuda(points, res_x, res_y, res_z, chunk_size, k, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z)

def pc_to_voxel_grid_chunk_long(points, res_x, res_y, res_z, chunk_size=256, k=1, num_keep=0, rmi_x=1.0, rmi_y=1.0, rmi_z=1.0, rma_x=1.0, rma_y=1.0, rma_z=1.0):
    """
    Convert a massive point cloud to a sparse voxel grid using spatial chunking and 64-bit indices.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        chunk_size: Size of the cubic chunk to process at one time to save VRAM (default: 256)
        k: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
        rmi_x, rmi_y, rmi_z: Minimum keeping ratio of the minimum bounding box
        rma_x, rma_y, rma_z: Maximum keeping ratio of the maximum bounding box

    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    return pc_to_voxel_grid_chunk_long_cuda(points, res_x, res_y, res_z, chunk_size, k, num_keep, rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z)

def pc2vg(points, res_x, res_y, res_z, chunk_size=None, k=1, num_keep=0, rmi_x=1.0, rmi_y=1.0, rmi_z=1.0, rma_x=1.0, rma_y=1.0, rma_z=1.0):
    """
    Unified Point Cloud to Voxel Grid function.
    Automatically selects between 32-bit/64-bit and standard/chunked kernels 
    based on the grid resolution and chunk_size parameter.
    
    Args:
        points: Tensor of shape (num_points, 3) containing point cloud coordinates
        res_x, res_y, res_z: Resolution (number of voxels) in each dimension
        chunk_size: Size of cubic chunk for VRAM savings. Set to 0 or None to use dense grid. (default: None)
        k: Minimum number of points required in a voxel to be considered occupied (default: 1)
        num_keep: Number of adjacent voxels to dilate and keep (default: 0)
        rmi_x, rmi_y, rmi_z: Minimum keeping ratio of the minimum bounding box
        rma_x, rma_y, rma_z: Maximum keeping ratio of the maximum bounding box

    Returns:
        voxel_grid: Tensor of shape (num_occupied_voxels, 3) containing coordinates of occupied voxels
        voxel_indices: Tensor of shape (num_occupied_voxels,) containing linear indices of occupied voxels
    """
    
    # Calculate maximum possible vertices in the dense grid
    max_dense_verts = (res_x + 1) * (res_y + 1) * (res_z + 1)
    
    # A 32-bit signed integer caps out at 2,147,483,647.
    # If our dense grid exceeds this, we MUST use the 64-bit (long) version to prevent overflow.
    use_long = max_dense_verts >= 2147483647
    
    # Decide whether to use spatial tiling (chunking)
    use_chunking = chunk_size is not None and chunk_size > 0
    
    # Route to the appropriate CUDA kernel
    if use_chunking:
        if use_long:
            return pc_to_voxel_grid_chunk_long_cuda(
                points, res_x, res_y, res_z, chunk_size, k, num_keep, 
                rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z
            )
        else:
            return pc_to_voxel_grid_chunk_cuda(
                points, res_x, res_y, res_z, chunk_size, k, num_keep, 
                rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z
            )
    else:
        if use_long:
            return pc_to_voxel_grid_long_cuda(
                points, res_x, res_y, res_z, k, num_keep, 
                rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z
            )
        else:
            return pc_to_voxel_grid_cuda(
                points, res_x, res_y, res_z, k, num_keep, 
                rmi_x, rmi_y, rmi_z, rma_x, rma_y, rma_z
            )
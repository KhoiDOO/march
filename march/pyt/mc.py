from .data import EDGE_TO_VERTICES, FACE_TABLE, INDEX
from .cons import *

import torch


def mc(
    grids: torch.Tensor,
    cubes: torch.LongTensor,
    values: torch.Tensor,
    iso: float,
) -> list[torch.FloatTensor, torch.LongTensor]:
    """
    Marching Cubes algorithm for extracting a mesh from a scalar field.

    Args:
        grids: A tensor of shape (N, 3) containing the coordinates of the grid points.
        cubes: A tensor of shape (M, 8) containing the indices of the cube vertices.
        values: A tensor of shape (N,) containing the scalar values at the grid points.
        iso: The isovalue for which to extract the mesh.
    Returns:
        A tuple containing:
            - A tensor of shape (K, 3) containing the vertex positions of the extracted mesh.
            - A tensor of shape (L, 3) containing the indices of the triangles in the extracted mesh.
    """
    
    device = grids.device
    
    edge2vertex = {} # (global_v1, global_v2) -> mesh_v_idx
    verts = [] # Global list of mesh vertex positions
    faces = [] # Global list of tri indices
    
    for idx, cube in enumerate(cubes):
        vs = values[cube]  # (8,)!
        
        cube_index = 0
        for i in range(8):
            if vs[INDEX[i]] < iso:
                cube_index |= 1 << i
        
        edge_indices = FACE_TABLE[cube_index]
        # cube is entirely in/out of the surface
        if len(edge_indices) == 0:
            continue
        
        tri = []
        tri_ps = []
        for i, edge in enumerate(edge_indices):
            vidx1, vidx2 = EDGE_TO_VERTICES[edge]
            vgidx1, vgidx2 = cube[vidx1].item(), cube[vidx2].item()
            
            eid = tuple(sorted((vgidx1, vgidx2)))
            
            if eid not in edge2vertex:
                
                p1, p2 = grids[vgidx1], grids[vgidx2]
                v1, v2 = vs[vidx1], vs[vidx2]
                
                p = None
                if abs(iso - v1) < EPS:
                    p = p1
                elif abs(iso - v2) < EPS:
                    p = p2
                elif abs(v1 - v2) < EPS:
                    p = p1
                elif p is None:
                    t = (iso - v1) / (v2 - v1)
                    p = p1 + t * (p2 - p1)
                
                edge2vertex[eid] = len(verts)
                verts.append(p)
            
            tri.append(edge2vertex[eid])
            tri_ps.append(verts[edge2vertex[eid]])
            
            if (i + 1) % 3 == 0:
                if not (
                    torch.allclose(tri_ps[0], tri_ps[1]) or
                    torch.allclose(tri_ps[1], tri_ps[2]) or
                    torch.allclose(tri_ps[2], tri_ps[0])
                ):
                    faces.append(tri)
                tri = []
                tri_ps = []
    
    if not verts:
        return torch.empty((0, 3), device=device), torch.empty((0, 3), dtype=torch.long, device=device)

    final_verts = torch.stack(verts) if torch.is_tensor(verts[0]) else torch.tensor(verts)
    final_faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    return final_verts.to(device), final_faces
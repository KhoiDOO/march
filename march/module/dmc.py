import torch
from torch import dtype, nn
import torch.nn.functional as F
from torch.autograd import Function

# Import the new DMC classes from your PyBind module
from .._C import DMCFI, DMCFL


class DMC(nn.Module):
    def __init__(self, vdtype=torch.float32, cdtype=torch.int32):
        super().__init__()
        self.vdtype = vdtype
        self.cdtype = cdtype
        
        # Route to the correct C++ backend based on index type
        if cdtype == torch.int32:
            dmc = DMCFI()
        elif cdtype == torch.int64:
            dmc = DMCFL()
        else:
            raise NotImplementedError(f"Unsupported dtype: {cdtype}")
            
        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, grid_vertices, voxels, values, iso, grid_colors):
                # Returns verts, quads [N, 4], and optional colors
                verts, quads, out_colors = dmc.forward(grid_vertices, voxels, values, iso, grid_colors)
                ctx.isovalue = iso
                ctx.save_for_backward(grid_vertices, values, grid_colors)
                return verts, quads, out_colors
            
            @staticmethod
            def backward(ctx, adj_verts, adj_quads, adj_out_colors):
                grid_vertices, values, grid_colors = ctx.saved_tensors
                iso = ctx.isovalue

                adj_values = torch.zeros_like(values)
                adj_grid_colors = None
                
                if grid_colors is not None and adj_out_colors is not None:
                    adj_grid_colors = torch.zeros_like(grid_colors)
                else:
                    # Pass None so PyBind correctly skips the color backward math
                    adj_grid_colors = None
                    adj_out_colors = None
                
                dmc.backward(
                    grid_vertices, 
                    values, 
                    adj_verts, 
                    adj_values, 
                    iso,
                    grid_colors,
                    adj_out_colors,
                    adj_grid_colors
                )
                
                # Return gradients matching the order of forward() arguments:
                # 1. grid_vertices -> None
                # 2. voxels -> None
                # 3. values -> adj_values
                # 4. iso -> None
                # 5. grid_colors -> adj_grid_colors
                return None, None, adj_values, None, adj_grid_colors
        
        self.func = DMCFunction
        self._dmc = dmc  # Keep reference alive
    
    def forward(self, grid_vertices, voxels, values, iso, grid_colors=None, triangulate=False):
        # Fast path: If the surface doesn't exist, return correctly shaped empty tensors
        if values.min() >= iso or values.max() <= iso:
            empty_verts = torch.zeros((0, 3), dtype=self.vdtype, device=grid_vertices.device)
            
            # Dynamically size the empty faces tensor based on the toggle
            face_dim = 3 if triangulate else 4
            empty_faces = torch.zeros((0, face_dim), dtype=self.cdtype, device=grid_vertices.device)
            
            if grid_colors is not None:
                empty_colors = torch.zeros((0, grid_colors.shape[-1]), dtype=self.vdtype, device=grid_vertices.device)
                return empty_verts, empty_faces, empty_colors
            return empty_verts, empty_faces
        
        # Run the C++ Forward Pass
        verts, quads, out_colors = self.func.apply(grid_vertices, voxels, values, iso, grid_colors)
        
        # if triangulate and quads.shape[0] > 0:
        #     # Split Quads [N, 4] into Triangles [2N, 3]
        #     # Triangle 1: [v0, v1, v2]
        #     tri1 = quads[:, [0, 1, 2]]
        #     # Triangle 2: [v0, v2, v3]
        #     tri2 = quads[:, [0, 2, 3]]
            
        #     # Concatenate them vertically. Shape becomes [N*2, 3]
        #     faces = torch.cat([tri1, tri2], dim=0)
            
        #     valid_faces_mask = (faces != -1).all(dim=1)
        #     faces = faces[valid_faces_mask]
        # else:
        #     faces = quads
            
        # if triangulate and quads.shape[0] > 0:
        #     # Find how many valid vertices exist in each quad
        #     valid_mask = (quads != -1)
        #     valid_counts = valid_mask.sum(dim=1)
            
        #     # -------------------------------------------------------------
        #     # 1. Process Full Quads (All 4 voxels exist)
        #     # -------------------------------------------------------------
        #     full_quads = quads[valid_counts == 4]
        #     tri1 = full_quads[:, [0, 1, 2]]
        #     tri2 = full_quads[:, [0, 2, 3]]
            
        #     # -------------------------------------------------------------
        #     # 2. Salvage Boundary Triangles (Only 3 voxels exist)
        #     # -------------------------------------------------------------
        #     partial_quads = quads[valid_counts == 3]
            
        #     # Locate exactly which slot contains the -1 sentinel
        #     empty_0 = partial_quads[:, 0] == -1
        #     empty_1 = partial_quads[:, 1] == -1
        #     empty_2 = partial_quads[:, 2] == -1
        #     empty_3 = partial_quads[:, 3] == -1
            
        #     # Salvage the remaining 3 vertices in cyclic order to preserve face normals
        #     salvaged_0 = partial_quads[empty_0][:, [1, 2, 3]]
        #     salvaged_1 = partial_quads[empty_1][:, [2, 3, 0]]
        #     salvaged_2 = partial_quads[empty_2][:, [3, 0, 1]]
        #     salvaged_3 = partial_quads[empty_3][:, [0, 1, 2]]
            
        #     # -------------------------------------------------------------
        #     # 3. Combine everything into a single watertight face tensor
        #     # -------------------------------------------------------------
        #     faces = torch.cat([tri1, tri2, salvaged_0, salvaged_1, salvaged_2, salvaged_3], dim=0)
            
        # else:
        #     faces = quads
        #     # If not triangulating, drop any quads that aren't perfectly complete
        #     faces = faces[(faces != -1).all(dim=1)]
        
        if triangulate and quads.shape[0] > 0:
            valid_mask = (quads != -1)
            valid_counts = valid_mask.sum(dim=1)
            
            # 1. Process Full Quads (All 4 voxels exist)
            full_quads = quads[valid_counts == 4]
            
            # --- THE SHORTEST DIAGONAL FIX ---
            # Get the 3D coordinates for all 4 corners of every quad
            v0 = verts[full_quads[:, 0]]
            v1 = verts[full_quads[:, 1]]
            v2 = verts[full_quads[:, 2]]
            v3 = verts[full_quads[:, 3]]
            
            # Calculate squared lengths of the two diagonals
            dist_02 = (v0 - v2).pow(2).sum(dim=-1)
            dist_13 = (v1 - v3).pow(2).sum(dim=-1)
            
            # Create a mask for quads where 0->2 is the shorter diagonal
            mask = dist_02 <= dist_13
            
            # Split along 0->2
            tri1_mask = full_quads[mask][:, [0, 1, 2]]
            tri2_mask = full_quads[mask][:, [0, 2, 3]]
            
            # Split along 1->3 for the warped quads
            tri1_not = full_quads[~mask][:, [0, 1, 3]]
            tri2_not = full_quads[~mask][:, [1, 2, 3]]
            
            # 2. Salvage Boundary Triangles (Your existing code)
            partial_quads = quads[valid_counts == 3]
            empty_0 = partial_quads[:, 0] == -1
            empty_1 = partial_quads[:, 1] == -1
            empty_2 = partial_quads[:, 2] == -1
            empty_3 = partial_quads[:, 3] == -1
            
            salvaged_0 = partial_quads[empty_0][:, [1, 2, 3]]
            salvaged_1 = partial_quads[empty_1][:, [2, 3, 0]]
            salvaged_2 = partial_quads[empty_2][:, [3, 0, 1]]
            salvaged_3 = partial_quads[empty_3][:, [0, 1, 2]]
            
            # 3. Combine everything
            faces = torch.cat([tri1_mask, tri2_mask, tri1_not, tri2_not, 
                               salvaged_0, salvaged_1, salvaged_2, salvaged_3], dim=0)
        
        if grid_colors is not None:
            return verts, faces.long(), out_colors

        return verts, faces.long()
import torch
from torch import dtype, nn
import torch.nn.functional as F
from torch.autograd import Function

from .._C import MCFI, MCFL


class MC(nn.Module):
    def __init__(self, vdtype=torch.float32, cdtype=torch.int32):
        super().__init__()
        self.vdtype = vdtype
        self.cdtype = cdtype
        if cdtype == torch.int32:
            mc = MCFI()
        elif cdtype == torch.int64:
            mc = MCFL()
        else:
            raise NotImplementedError(f"Unsupported dtype: {cdtype}")
            
        class MCFunction(Function):
            @staticmethod
            def forward(ctx, grid_vertices, voxels, values, iso, grid_colors):
                verts, tris, out_colors = mc.forward(grid_vertices, voxels, values, iso, grid_colors)
                ctx.isovalue = iso
                ctx.save_for_backward(grid_vertices, values, grid_colors)
                return verts, tris, out_colors
            
            @staticmethod
            def backward(ctx, adj_verts, adj_faces, adj_out_colors):
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
                
                mc.backward(
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
        
        self.func = MCFunction
        self._mc = mc  # Keep reference alive
    
    def forward(self, grid_vertices, voxels, values, iso, grid_colors=None):
        if values.min() >= iso or values.max() <= iso:
            empty_verts = torch.zeros((0, 3), dtype=self.vdtype, device=grid_vertices.device)
            empty_tris = torch.zeros((0, 3), dtype=self.cdtype, device=grid_vertices.device)
            
            if grid_colors is not None:
                empty_colors = torch.zeros((0, grid_colors.shape[-1]), dtype=self.vdtype, device=grid_vertices.device)
                return empty_verts, empty_tris, empty_colors
            return empty_verts, empty_tris
        
        verts, tris, out_colors = self.func.apply(grid_vertices, voxels, values, iso, grid_colors)
        
        if grid_colors is not None:
            return verts, tris.long(), out_colors

        return verts, tris.long()

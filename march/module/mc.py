import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from .._C import MCF


class DMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            mc = MCF()
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")
            
        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, grid_vertices, cubes, values, iso):
                verts, tris = mc.forward(grid_vertices, cubes, values, iso)
                ctx.isovalue = iso
                ctx.save_for_backward(grid_vertices, values)
                return verts, tris
            
            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                grid_vertices, values = ctx.saved_tensors
                iso = ctx.isovalue

                adj_values = torch.zeros_like(values)
                mc.backward(grid_vertices, values, adj_verts, adj_values, iso)
                return None, None, adj_values, None
        
        self.func = DMCFunction
        self._mc = mc  # Keep reference alive
    
    def forward(self, grid_vertices, cubes, values, iso):
        if values.min() >= iso or values.max() <= iso:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid_vertices.device), \
                torch.zeros((0, 3), dtype=torch.int32, device=grid_vertices.device)
        
        verts, tris = self.func.apply(grid_vertices, cubes, values, iso)

        return verts, tris.long()

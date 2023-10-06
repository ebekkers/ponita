import torch


def invariant_attr_r3s2_spatial(pos, grid, edge_index = None):
    if edge_index is None:
        # Assuming shapes pos: [B, X, 3], grid: [B, O, 3]
        pos_source, pos_target = pos[:,:,None,:], pos[:,None,:,:]
        rel_pos = pos_source - pos_target
        invariant1 = (rel_pos[:,:,:,None,:] * grid[:,None,None,:,:]).sum(dim=-1,keepdim=True)
        invariant2 = (rel_pos[:,:,:,None,:] - grid[:,None,None,:,:] * invariant1).norm(dim=-1, keepdim=True)
    else:
        pos_source, pos_target = pos[edge_index[0]], pos[edge_index[1]]
        grid = grid[edge_index[0]] # Copy grid to each edge
        rel_pos = (pos_source - pos_target).unsqueeze(dim=-2)  # [BX, 1, 3]
        invariant1 = (rel_pos * grid).sum(dim=-1, keepdim=True)  # [BX, N, 1]
        invariant2 = (rel_pos - invariant1 * grid).norm(dim=-1, keepdim=True)   
    return torch.cat([invariant1, invariant2 - 3.0],dim=-1)

def invariant_attr_r3s2_spherical(grid):
    # Assuming shape grid: [B, O, 3]
    # Compute inner products
    invariant3 = (grid[:,:,None,:] * grid[:,None,:,:]).sum(dim=-1, keepdim=True)
    # We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear 
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))
    return invariant3

def invariant_attr_r3(pos, edge_index=None):
    if edge_index is None:
        return torch.norm(pos[:,:,None,:]-pos[:,None,:,:], dim=-1, keepdim=True)
    else:
        return torch.norm(pos[edge_index[0]]-pos[edge_index[1]], dim=-1, keepdim=True)
import torch


def invariant_attr_r3(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]] # [num_edges, 3]
    dists = (pos_send - pos_receive).norm(dim=-1, keepdim=True)    # [num_edges, 1]
    return dists


def invariant_attr_r3s2(pos, ori_grid, edge_index, separable=False):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                # [num_edges, 3]
    rel_pos = (pos_send - pos_receive)                                            # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]                                                 # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None,:,:]                                               # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None,:]                                              # [num_ori, 1, 3]

    invariant1 = (rel_pos * ori_grid_a).sum(dim=-1, keepdim=True)                 # [num_edges, num_ori, 1]
    invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(dim=-1, keepdim=True)   # [num_edges, num_ori, 1]
    invariant3 = (ori_grid_a * ori_grid_b).sum(dim=-1, keepdim=True)              # [num_ori, num_ori, 1]
    
    # Note: We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear 
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))
    
    if separable:
        return torch.cat([invariant1, invariant2],dim=-1), invariant3             # [num_edges, num_ori, 2], [num_ori, num_ori, 1]
    else:
        invariant1 = invariant1[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant2 = invariant2[:,:,None,:].expand(-1,-1,ori_grid.shape[0],-1)    # [num_edges, num_ori, num_ori, 1]
        invariant3 = invariant3[None,:,:,:].expand(invariant1.shape[0],-1,-1,-1)  # [num_edges, num_ori, num_ori, 1]
        return torch.cat([invariant1, invariant2, invariant3],dim=-1)             # [num_edges, num_ori, num_ori, 3]
    
    

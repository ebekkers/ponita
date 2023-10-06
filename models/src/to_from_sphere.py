import torch


def vec_to_sphere(vec, grid):
    if len(vec.shape)==3:
        return torch.einsum('bcd,bnd->bnc', vec, grid)  # graph mode
    else:
        return torch.einsum('bxcd,bnd->bxnc', vec, grid)  # dense mode
    
def scalar_to_sphere(scalar, grid):
    return scalar.unsqueeze(-2).repeat_interleave(grid.shape[-2], dim=-2)

def sphere_to_vec(spherical_signal, grid):
    if len(spherical_signal.shape)==3:
        return (spherical_signal[:,:,None,:] * grid[:,:,:,None]).mean(dim=1)
    else:
        return torch.mean(grid[:,None,:,:] * spherical_signal, dim=-2)

def sphere_to_scalar(spherical_signal):
    return spherical_signal.mean(dim=-2)
import torch
import torch_geometric


class SeparableConvR3S2(torch_geometric.nn.MessagePassing):
    """
    """
    def __init__(self, hidden_dim, basis_dim, bias=True, aggr="add"):
        super().__init__(node_dim=0, aggr=aggr)
        self.kernel_spatial = torch.nn.Linear(basis_dim, hidden_dim, bias=False)
        self.kernel_spherical = torch.nn.Linear(basis_dim, hidden_dim, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(hidden_dim))
            self.bias.data.zero_()
        else:
            self.register_parameter('bias', None)
        self.register_buffer("callibrated", torch.tensor(False))
        
    def forward(self, x, edge_attr_spatial, edge_attr_sphere, edge_index):
        """
        """
        # Sample the convolution kernels
        kernel_spatial = self.kernel_spatial(edge_attr_spatial)     # [num_edges, num_ori, num_channels]
        kernel_spherical = self.kernel_spherical(edge_attr_sphere)  # [num_ori, num_ori, num_channels]

        # Do the convolutions: 1. Spatial conv, 2. Spherical conv
        x_1 = self.propagate(edge_index, x=x, kernel_spatial=kernel_spatial)
        x_2 = torch.einsum('boc,opc->bpc', x_1, kernel_spherical) / kernel_spherical.shape[-2]

        # Re-callibrate the initializaiton
        if self.training and not(self.callibrated):
            self.callibrate(x.std(), x_1.std(), x_2.std())

        # Add bias
        if self.bias is not None:
            return x_2 + self.bias
        else:  
            return x_2

    def message(self, x_j, kernel_spatial):
        out = kernel_spatial * x_j  # [B, S, C] * [B, S, C], S = num elements in source fiber
        return out
    
    def callibrate(self, std_in, std_1, std_2):
        print('Callibrating...')
        with torch.no_grad():
            self.kernel_spatial.weight.data = self.kernel_spatial.weight.data * std_in/std_1
            self.kernel_spherical.weight.data = self.kernel_spherical.weight.data * std_1/std_2
            self.callibrated = ~self.callibrated


class ConvNextR3S2(torch.nn.Module):
    """
    """
    def __init__(self, feature_dim, attr_dim, act=torch.nn.GELU(), layer_scale=1e-6, widening_factor=4): 
        super().__init__()

        self.conv = SeparableConvR3S2(feature_dim, attr_dim)
        self.act_fn = act
        self.linear_1 = torch.nn.Linear(feature_dim, widening_factor * feature_dim)
        self.linear_2 = torch.nn.Linear(widening_factor * feature_dim, feature_dim)
        if layer_scale is not None:
            self.layer_scale = torch.nn.Parameter(torch.ones(feature_dim) * layer_scale)
        else:
            self.register_buffer('layer_scale', None)
        self.norm = torch.nn.LayerNorm(feature_dim)

    def forward(self, x, basis_spatial, basis_spherical, edge_index=None):
        """
        """
        input = x
        x = self.conv(x, basis_spatial, basis_spherical, edge_index)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        if input.shape == x.shape: 
            x = x + input
        return x
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch.nn import LazyBatchNorm1d
from ponita.utils.to_from_sphere import sphere_to_scalar, sphere_to_vec
from ponita.nn.convnext import ConvNextR3S2
from ponita.nn.embedding import PolynomialFeatures
from ponita.utils.windowing import PolynomialCutoff
from ponita.transforms import PositionOrientationGraph, SEnInvariantAttributes
from torch_geometric.transforms import Compose


class PONITA(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 output_dim_vec = 0,
                 radius = 1000.,
                 n=20,
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 task_level='graph',
                 multiple_readouts=True):
        super().__init__()

        # Input output settings
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level=='graph'

        # For constructing the position-orientation graph and its invariants
        self.transform = Compose([PositionOrientationGraph(n), SEnInvariantAttributes(separable=True)])

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Kernel basis functions and spatial window
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        self.basis_fn = nn.Sequential(PolynomialFeatures(degree), nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        self.basis_fn_ori = nn.Sequential(PolynomialFeatures(degree), nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        self.windowing_fn = PolynomialCutoff(radius)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim, hidden_dim, False)
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            self.interaction_layers.append(ConvNextR3S2(hidden_dim, basis_dim, act=act_fn, widening_factor=widening_factor, layer_scale=layer_scale))
            if multiple_readouts or i == (num_layers - 1):
                self.read_out_layers.append(nn.Linear(hidden_dim, output_dim + output_dim_vec))
            else:
                self.read_out_layers.append(None)
    
    def forward(self, graph):

        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists).unsqueeze(-2)
        kernel_basis_ori = self.basis_fn_ori(graph.attr_ori)

        # Initial feature embeding
        x = self.x_embedder(graph.f)

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, kernel_basis, kernel_basis_ori, graph.edge_index)
            if readout_layer is not None: readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)
        
        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)
        
        # Read out scalar and vector predictions
        output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch)
        output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar, output_vector
    
    def scalar_readout_fn(self, readout_scalar, batch):
        if self.output_dim > 0:
            output_scalar = sphere_to_scalar(readout_scalar)
            if self.global_pooling:
                output_scalar=global_add_pool(output_scalar, batch)
        else:
            output_scalar = None
        return output_scalar
    
    def vec_readout_fn(self, readout_vec, ori_grid, batch):
        if self.output_dim_vec > 0:
            output_vector = sphere_to_vec(readout_vec, ori_grid)
            if self.global_pooling:
                output_vector = global_add_pool(output_vector, batch)
        else:
            output_vector = None
        return output_vector

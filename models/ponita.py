import torch
import torch.nn as nn
from src.invariants import invariant_attr_r3, invariant_attr_r3s2_spatial, invariant_attr_r3s2_spherical
from src.to_from_sphere import scalar_to_sphere, sphere_to_scalar, sphere_to_vec, vec_to_sphere
from src.convnext import ConvNextR3S2
from src.embedding import PolynomialFeatures
from src.rotation import uniform_grid_s2, random_matrix
from src.windowing import PolynomialCutoff
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool


class PONITA(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 input_dim_vec = 0,
                 output_dim_vec = 0,
                 radius = 1000.,
                 n=20,
                 M='R3S2',
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 task_level='graph'):
        super().__init__()

        # Input output settings
        self.input_dim, self.input_dim_vec = input_dim, input_dim_vec
        self.output_dim, self.output_dim_vec = output_dim, output_dim_vec
        self.global_pooling = task_level=='graph'
        self.num_layers = num_layers

        # Activation function to use internally
        act_fn = torch.nn.GELU()

        # Geometric setup
        if M=="R3":  # TODO: Add the R3 model to this code base
            self.grid = None
        elif M=="R3S2":
            self.register_buffer('grid', uniform_grid_s2(n))  # [n,d]

        # Attribute embedding and basis functions
        basis_dim = hidden_dim if (basis_dim is None) else basis_dim
        attr_embed_fn_spatial = PolynomialFeatures(degree)  # TODO: Test against RFFs
        self.basis_fn_spatial = nn.Sequential(attr_embed_fn_spatial, nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)
        attr_embed_fn_spherical = PolynomialFeatures(degree)  # TODO: Test against RFFs
        self.basis_fn_spherical = nn.Sequential(attr_embed_fn_spherical, nn.LazyLinear(hidden_dim), act_fn, nn.Linear(hidden_dim, basis_dim), act_fn)

        # Windowing function to localize the kernel
        self.windowing_fn = PolynomialCutoff(radius)

        # Initial node embedding
        self.x_embedder = nn.Linear(input_dim + input_dim_vec, hidden_dim, bias=False)
        
        # Make feedforward network
        self.interaction_layers = nn.ModuleList()
        self.read_out_layers = nn.ModuleList()
        for i in range(num_layers):
            self.interaction_layers.append(ConvNextR3S2(hidden_dim, basis_dim, act=act_fn, widening_factor=widening_factor, layer_scale=layer_scale))
            self.read_out_layers.append(nn.Linear(hidden_dim, output_dim + output_dim_vec))
    
    def forward(self, graph):

        # Generate spherical grids
        batch = graph.batch
        batch_size = batch.max() + 1
        edge_index = graph.edge_index if hasattr(graph, "edge_index") else None

        # Randomly rotate the grids (each graph gets its own grid)
        rand_SO3 = random_matrix(batch_size, device=graph.batch.device)  # random rotation for each graph in the batch
        grid = torch.einsum('bij,nj->bni', rand_SO3, self.grid)  # rotate grid for each graph in the batch
        node_grid = grid if edge_index is None else grid[batch]  # Copy the grid to each node (when in graph mode)
        
        # Read in the input scalars and vectors and switch to dense mode if no edge_index is provided
        inputs = []
        if edge_index is None:
            # Then switch to fully connected dense mode
            pos, mask = to_dense_batch(graph.pos, graph.batch)
            if self.input_dim > 0:
                inputs.append(scalar_to_sphere(to_dense_batch(graph.x, graph.batch)[0], node_grid))
            if self.input_dim_vec > 0:
                inputs.append(vec_to_sphere(to_dense_batch(graph.vec, graph.batch)[0], node_grid))
        else:
            # Otherwise to torch geometric style graph mode (requiring edge_index)
            pos = graph.pos
            if self.input_dim > 0:
                inputs.append(scalar_to_sphere(graph.x, node_grid))
            if self.input_dim_vec > 0:
                inputs.append(vec_to_sphere(graph.vec, node_grid))
        x = torch.cat(inputs, dim=-1) 

        # Compute invariants and sample the basis functions (shared over all layers)
        attr_spatial = invariant_attr_r3s2_spatial(pos, node_grid, edge_index=edge_index)
        attr_spherical = invariant_attr_r3s2_spherical(node_grid)
        kernel_basis_spatial = self.basis_fn_spatial(attr_spatial)
        kernel_basis_spherical = self.basis_fn_spherical(attr_spherical)
    
        # Window the spatial basis functions as to localize the kernels
        dists = invariant_attr_r3(pos, edge_index=edge_index)
        attr_spatial = attr_spatial * self.windowing_fn(dists).unsqueeze(-2)

        # Initial feature embeding
        x = self.x_embedder(x)

        # Interaction + readout layers
        readout = 0.
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, kernel_basis_spatial, kernel_basis_spherical, edge_index)
            readout += readout_layer(x) / self.num_layers
        
        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)
        # Converting back to graph mode
        if edge_index is None:  
            readout_scalar = readout_scalar[mask]  
            readout_vec = readout_vec[mask]
            node_grid = node_grid[batch]
        # Read out scalars
        if self.output_dim > 0:
            output_scalar = sphere_to_scalar(readout_scalar)
            if self.global_pooling:
                output_scalar=global_add_pool(output_scalar, batch)
        else:
            output_scalar = None
        # Read out vectors
        if self.output_dim_vec > 0:
            output_vector = sphere_to_vec(readout_vec, node_grid)
            if self.global_pooling:
                output_vector = global_add_pool(output_vector, batch)
        else:
            output_vector = None

        # Return scalar and vector prediction
        return output_scalar, output_vector

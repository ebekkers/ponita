import torch
import torch.nn as nn
from ponita.models.ponita import Ponita
from ponita.models.ponita import PonitaFiberBundle

class TemporalPonita(PonitaFiberBundle):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 args,
                 output_dim_vec=0,
                 radius=None,
                 num_ori=20,
                 basis_dim=None,
                 degree=3,
                 widening_factor=4,
                 layer_scale=None,
                 task_level='graph',
                 multiple_readouts=True,
                 **kwargs):
        
        super().__init__(input_dim,
                         hidden_dim,
                         output_dim,
                         num_layers,
                         output_dim_vec=output_dim_vec,
                         radius=radius,
                         num_ori=num_ori,
                         basis_dim=basis_dim,
                         degree=degree,
                         widening_factor=widening_factor,
                         layer_scale=layer_scale,
                         task_level=task_level,
                         multiple_readouts=multiple_readouts,
                         **kwargs)  
        
        self.args = args
        self.conv1d_layer = nn.ModuleList()
        self.kernel_size = 3
        self.stride =  5
        self.conv1d_layer.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, groups = hidden_dim,  kernel_size=self.kernel_size, stride=self.stride))
        


        
        

    def forward(self, graph):

        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists).unsqueeze(-2)
        fiber_kernel_basis = self.fiber_basis_fn(graph.fiber_attr)


        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Given this, do we actually need temporal edges? Is it not just an ordered smoshing of rows in x?
        # What about positions, how are they considered and how does it matter?
        ''' If we assume that we only keep the spatial edges, do we need to do really anything special?
        '''

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            # Assuming only spatial edges
            print('x', x.shape)
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, fiber_attr=fiber_kernel_basis, batch=graph.batch)
            print('x', x.shape)
            # Do we run a temporal convolution here?
            
            x = self.conv1d(x, graph)
            if readout_layer is not None: readouts.append(readout_layer(x))

        readout = sum(readouts) / len(readouts)

        # Read out the scalar and vector part of the output
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)
        
        # Read out scalar and vector predictions
        output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch)
        output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar, output_vector
      
    
    def conv1d(self, x, graph):
        """ Perform 1D convolution on the time axis 

        This would need to keep trac of the vid_id, and only perform convolutions within the same vid_id


        """
        num_land_marks = self.args.n_nodes
        #print('num landmarks', num_land_marks)
        batch_size = len(graph.batch.unique())
        #print('batch sizs', batch_size)
        num_nodes_batch, num_ori, num_channels = x.shape
        #print('num nodes batch', num_nodes_batch)
        #print('num ori', num_ori)
        #print('num channels', num_channels)
        print('x in', x.shape)
        x = x.view(-1, batch_size*num_land_marks*num_ori, num_channels)
        #print('x view', x.shape)
        x = x.permute(1, 2, 0)
        #print('x permute', x.shape, x.device)
        # this should be in the init func tno?
        for layer in self.conv1d_layer:
            x = layer(x)
        #print('out', x.shape)
        x = x.permute(2, 0, 1)
        #print('x permute out', x.shape)
        #print('reshape size', num_nodes_batch - (self.kernel_size-1)*num_land_marks)
        downsample = x.shape[0]*num_land_marks
        x = x.reshape(downsample, num_ori, num_channels)
        #x = x.view(num_nodes_batch - (self.kernel_size-1)*num_land_marks, num_ori, num_channels)
        print('x out', x.shape)
        # unsmooshh out again by doing reverse of the transformations we did before the 1d conv
        return x


if __name__ == "__main__":
    
    time_ponita = TemporalPonita(27, 64, 10, 4)
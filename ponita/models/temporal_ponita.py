import torch
import torch.nn as nn
from ponita.models.ponita import PonitaFiberBundle
from ponita.nn.conv import Conv, FiberBundleConv
from ponita.nn.convnext import ConvNext
import math

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
        self.num_layers = num_layers    
        self.kernel_size = self.args.kernel_size
        self.stride =  self.args.stride
        self.padding = int((self.kernel_size - 1) / 2)
        self.conv1d_layer.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, groups = hidden_dim,  kernel_size=self.kernel_size, stride=self.stride, padding = self.padding))
        
        self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
        self.n_edges = len(self.inward_edges)
        
        dropout_rate = 0.1
        self.dropout = nn.Dropout(dropout_rate)

        # Tot edges per frame is number of nodes (self.edges) +  number of edges (except in the last frame )
        # Tot edges is (self.n_self_edges - 1), last frame does not have a self edge, plus number of edges* number of frames 
        self.tot_edges = args.n_nodes + self.n_edges
        
    def calc_1dconv_output_shape(self, input_shape):
        """ Calculate the output shape of a 1D convolution """
        return math.floor((input_shape - self.kernel_size)/self.stride + 1) 

    def forward(self, graph):

        # Lift and compute invariants
        graph = self.transform(graph)

        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(graph.attr) * self.windowing_fn(graph.dists).unsqueeze(-2)
        fiber_kernel_basis = self.fiber_basis_fn(graph.fiber_attr)

        # Initial feature embeding
        x = self.x_embedder(graph.x)

        # Interaction + readout layers
        readouts = []

        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            y = x

            # Assuming only spatial edges
            
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, fiber_attr=fiber_kernel_basis, batch=graph.batch)
                
            x, graph = self.conv1d(x, graph)

            if readout_layer is not None: 
                
                readouts.append(readout_layer(x))
        
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
        
        x_conv = []
        start_idx = 0
        for n_frames in graph.n_frames:
            # Select range corresponding to graph (n_frames x n_nodes)
            n_idx = n_frames*num_land_marks
            x_tmp = x[start_idx:start_idx+n_idx,]

            # Rearrange tensor
            num_nodes_batch, num_ori, num_channels = x_tmp.shape
            
            x_tmp = x_tmp.view(-1, num_land_marks*num_ori, num_channels)
            x_tmp = x_tmp.permute(1, 2, 0)

            # Convolution is performed on the last axis of the input data 
            for layer in self.conv1d_layer:
                x_tmp = layer(x_tmp)
                x_tmp = self.dropout(x_tmp)

            x_tmp = x_tmp.permute(2, 0, 1)

            #downsample = x.shape[0]*num_land_marks
            #x = x.reshape(downsample, num_ori, num_channels)
            x_tmp = x_tmp.reshape(num_nodes_batch, num_ori, num_channels)
            x_conv.append(x_tmp)

            # Update frame index
            start_idx += n_frames*num_land_marks
        
        x = torch.cat(x_conv, dim=0)

        #graph.edge_index = graph.edge_index[:,]
        
       

        # To print the edge index transformation
        # TODO: fix this for stride
        # Next problem is a position component, how do we select the position
        
        return x, graph

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TCNUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        use_drop=True,
        drop_size=1.92,
        num_points=25,
        block_size=41,
    ):
        super(TCNUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0,pad),
            stride=(1, stride),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.use_drop = False
        if use_drop:
            self.dropT = DropGraphTemporal(block_size=block_size)

    def forward(self, x, keep_prob=None, A=None):
        # In dim, out dim, n_frames, nodes 
        x = self.bn(self.conv(x))
        if self.use_drop:
            x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class DropGraphTemporal(nn.Module):
    def __init__(self, block_size=7):
        super(DropGraphTemporal, self).__init__()
        self.block_size = block_size

    def forward(self, x, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return x

        n, c, t, v = x.size()

        input_abs = torch.mean(torch.mean(torch.abs(x), dim=3), dim=1).detach()
        input_abs = (input_abs / torch.sum(input_abs) * input_abs.numel()).view(n, 1, t)
        gamma = (1.0 - self.keep_prob) / self.block_size
        input1 = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1, c * v, 1)
        m_sum = F.max_pool1d(
            M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2
        )
        mask = (1 - m_sum).to(device=m_sum.device, dtype=m_sum.dtype)
        return (
            (input1 * mask * mask.numel() / mask.sum())
            .view(n, c, v, t)
            .permute(0, 1, 3, 2)
        )


if __name__ == "__main__":
    
    time_ponita = TemporalPonita(27, 64, 10, 4)
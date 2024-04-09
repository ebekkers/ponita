import torch
import torch.nn as nn
from ponita.models.ponita import PonitaFiberBundle
from ponita.nn.conv import Conv, FiberBundleConv
from ponita.nn.convnext import ConvNext
import torch.nn.functional as F
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
        #self.conv1d_layer = nn.ModuleList()
        self.num_layers = num_layers    
        self.kernel_size = self.args.kernel_size
        self.stride =  self.args.stride
        self.padding = int((self.kernel_size - 1) / 2)
        #self.conv1d_layer.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, groups = hidden_dim,  kernel_size=self.kernel_size, stride=self.stride, padding = self.padding))
        
        self.inward_edges = [[2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
        self.n_edges = len(self.inward_edges)
        
        self.tconv = TCNUnit(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding, dropout_rate = args.temporal_dropout_rate)

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
            x_tmp = self.tconv.forward(x_tmp)

            x_tmp = x_tmp.permute(2, 0, 1)
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
        dropout_rate=0.1,
        num_points=25,
        padding = 0,
        block_size=41,
        
    ):
        super(TCNUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels)
        conv_init(self.conv)
        self.in_channels = in_channels
        self.out_channels = out_channels

        #self.attention = nn.Conv1d(out_channels, 1, kernel_size=kernel_size, padding=pad)
        #nn.Conv1d(out_channels, kernel_size=kernel_size, padding = pad)  # Produces a single attention score per time step
        #nn.init.xavier_uniform_(self.attention.weight)
        #nn.init.constant_(self.attention.weight, 0)
        #nn.init.zeros_(self.attention.bias)
        self.dropout = nn.Dropout(dropout_rate)
        #self.bn = nn.BatchNorm1d(out_channels)
        #bn_init(self.bn, 1)
        self.attention = TAttnUnit(in_channels)
        

    def forward(self, x):
        x = self.attention.forward(x)
        x = self.dropout(x)
        y = self.conv(x)
        return y 



class TAttnUnit(nn.Module):
    def __init__(
        self,
        hid_dim
    ):
        super(TAttnUnit, self).__init__()
        self.hidden_dim = hid_dim
        
    def invariant(self, x):
        return x
    
    def inv_emb_to_q(self, inv):
        q = inv
        return q
    
    def inv_emb_to_k(self, inv):
        k = inv
        return k
    
    def x_to_v(self, x):
        # Interpreting x as values (appearances, high freq info)
        return x

    def forward(self, x):
        # Assuming x's shape: [number of features, batch size, time axis]
        # First, ensure x is correctly permuted: [batch size, number of features, time axis]

        #x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        # Get invariants (if any transformation is needed, it's done here)
        inv = self.invariant(x)
        
        # Calculate query, key, value
        q = self.inv_emb_to_q(inv)  # [batch size, number of features, time axis]
        k = self.inv_emb_to_k(inv)
        v = self.x_to_v(x)
        q = q.permute(1, 2, 0)
        k = k.permute(1, 2, 0)

        

        # Compute attention scores: [batch size, time axis, time axis]
        # [FEATURES, batch_size, time axis]
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Softmax to obtain probabilities
        attn_probs = F.softmax(scores, dim=-1)
        
        # Apply attention to values (v needs to be [batch size, time axis, number of features] too)
        v = v.permute(1, 2, 0)
        y = torch.matmul(attn_probs, v)  # [batch size, time axis, number of features]
        
        
        # Optionally, transpose back if required: [number of features, batch size, time axis]
        y = y.permute(0, 2, 1)

        return y
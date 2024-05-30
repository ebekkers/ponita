import torch
import torch.nn as nn
from ponita.models.ponita import PonitaFiberBundle
from ponita.nn.attn import TAttnUnit


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
        
        self.num_layers = num_layers
        self.args = args
        self.num_land_marks = self.args.n_nodes    
        self.kernel_size = self.args.kernel_size
        self.stride =  self.args.stride
        self.padding = int((self.kernel_size - 1) / 2)
        
        self.tconv = TCNUnit(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=self.kernel_size, stride=self.stride, padding = self.padding, dropout_rate = args.temporal_dropout_rate)

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
            
            # Perform spatial convolutions
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
            x = interaction_layer(x, graph.edge_index, edge_attr=kernel_basis, fiber_attr=fiber_kernel_basis, batch=graph.batch)
            
            # Perform temporal convolution
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]    
            x = self.conv1d(x, graph)
            
            # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
            if readout_layer is not None: 

                # x [n_frames_per_vid x n_nodes x batch_size, num_ori, hidden_dim]
                # Pool across spatial temporal graph
                #x_ = self.TS_Pooling(x, graph)

                # x [batch_size, num_ori, hidden_dim]
                x_ = readout_layer(x)
                
                # x [n_frames_per_vid x n_nodes x batch_size, num_ori, n_classes]
                readouts.append(x_)
        
        readout = sum(readouts) / len(readouts)

        
        # readout_scalar [n_frames_per_vid x n_nodes x batch_size, num_ori, n_classes]
        readout_scalar, readout_vec = torch.split(readout, [self.output_dim, self.output_dim_vec], dim=-1)

        # output scalar [batch_size, n_classes]
        output_scalar = self.scalar_readout_fn(readout_scalar, graph.batch)
        
        # This is none
        output_vector = self.vec_readout_fn(readout_vec, graph.ori_grid, graph.batch)

        # Return predictions
        return output_scalar, output_vector
      
    def TS_Pooling(self, x, graph):
        """ Perform temporal pooling on the time axis """
        x_agg = []
        start_idx = 0
        for n_frames in graph.n_frames:
            # Select range corresponding to graph (n_frames x n_nodes)
            n_idx = n_frames*self.num_land_marks
            x_tmp = x[start_idx:start_idx+n_idx,]

            # Rearrange tensor
            num_nodes_batch, num_ori, num_channels = x_tmp.shape

            # [N_frames, num_landmarks, num_ori, num_channels]
            x_tmp = x_tmp.view(-1, self.num_land_marks, num_ori, num_channels)

            # Aggregate spatial dimension 
            x_tmp = x_tmp.mean(dim=1)

            # Aggregate temporal dimension 
            x_tmp = x_tmp.mean(dim=0)
            x_agg.append(x_tmp)
        return x_agg
    
    
    
    def conv1d(self, x, graph):
        """ Perform 1D convolution on the time axis 
        """

        x_conv = []
        start_idx = 0

        # For each graph in the batch
        for n_frames in graph.n_frames:
            
            # Select range corresponding to graph (n_frames x n_nodes)
            n_idx = n_frames*self.num_land_marks
            # TODO: Should this be n_frames or n_frames-1
            x_tmp = x[start_idx:start_idx+n_idx,]

            # Rearrange tensor
            # TODO: Check view vs permute vs restack
            num_nodes_batch, num_ori, num_channels = x_tmp.shape

            # [N_frames, num_landmarks*num_ori, num_channels]
            x_tmp = x_tmp.view(-1, self.num_land_marks*num_ori, num_channels)

            # [num_landmarks*num_ori, num_channels, N_frames]
            x_tmp = x_tmp.permute(1, 2, 0)

            # Convolution is performed on the last axis of the input data 
            x_tmp = self.tconv.forward(x_tmp)

            # Reshape back to original input shape
            x_tmp = x_tmp.permute(2, 0, 1)
            x_tmp = x_tmp.reshape(num_nodes_batch, num_ori, num_channels)
            x_conv.append(x_tmp)

            # Update frame indexing
            start_idx += n_frames*self.num_land_marks
        
        x = torch.cat(x_conv, dim=0)
        
        return x

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
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, groups=in_channels)
        self.conv_init(self.conv)
        self.conv_init(self.conv1)
        self.in_channels = in_channels
        self.out_channels = out_channels

        #self.attention = nn.Conv1d(out_channels, 1, kernel_size=kernel_size, padding=pad)
        #nn.Conv1d(out_channels, kernel_size=kernel_size, padding = pad)  # Produces a single attention score per time step
        #nn.init.xavier_uniform_(self.attention.weight)
        #nn.init.constant_(self.attention.weight, 0)
        #nn.init.zeros_(self.attention.bias)
        #self.dropout = nn.Dropout(dropout_rate)
        #self.bn = nn.BatchNorm1d(out_channels)
        #bn_init(self.bn, 1)
        #self.attention = TAttnUnit(out_channels)

        # self.activation = nn.ReLU()
        self.activation = nn.GELU()
    
    def conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

    
    def forward(self, x):
        
        #x = self.attention.forward(x)
        y = self.conv(x)
        
        # Use an activation function 
        y = self.activation(y)

        y = self.conv1(y)
        y = self.activation(y)

        # Residual connection 
        return y + x
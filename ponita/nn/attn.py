import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TAttnUnit(nn.Module):
    def __init__(
        self,
        hid_dim
    ):
        super(TAttnUnit, self).__init__()
        self.hidden_dim = 32*18*27
        # Define linear transformations 
        # for Q, K, and V
        self.q_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)
        self.k_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)
        self.v_transform = nn.Linear( self.hidden_dim,  self.hidden_dim, bias=False)
        
        # Temporal encoding
        self.max_time_steps = 240

        # Not sure about this?
        #self.temporal_encoding = nn.Parameter(torch.randn(1, self.max_time_steps, hid_dim))
    
    def conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
        nn.init.constant_(conv.bias, 0)

                
    def invariant(self, x):
        return x
    
    def inv_emb_to_q(self, inv):
        q = inv
        return self.q_transform(q)
    
    def inv_emb_to_k(self, inv):
        k = inv
        return self.k_transform(k)
    
    def x_to_v(self, x):
        # Interpreting x as values (appearances, high freq info)
        return self.v_transform(x)

    def forward(self, x):
        # x in : [num_landmarks x num_orientations, num_channels, num_frames]

        # x: [num_landmarks x num_orientations, num_frames, num_channels]
        x = x.permute(0, 2, 1) 


        
        # q, v, k: [num_landmarks x num_orientations, num_frames, num_channels]
        q = self.q_transform(x)
        k = self.k_transform(x)
        v = self.v_transform(x)
        
        d_k = q.size(1)
        # scores: [num_landmarks x num_orientations, num_frames x num_frames]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  
        attn_probs = F.softmax(scores, dim=-1)
       
       # y: [num_landmarks x num_orientations, num_frames, hid_dim]
        y = torch.matmul(attn_probs, v)  
        
        # y: [num_landmarks x num_orientations, num_channels, num_frames]
        y = y.permute(0, 2, 1)  
       
        return y
    
    def time_encoding(self, x):
        time_steps = x.size(1)
        temporal_encoding = self.temporal_encoding[:,:time_steps,:]
        x = x + temporal_encoding
        return x 
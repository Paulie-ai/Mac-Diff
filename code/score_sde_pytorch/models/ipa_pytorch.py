"""Fork of Openfold's IPA."""

import torch
import torch.nn as nn
import math
from typing import Optional, Callable, List, Sequence
from scipy.stats import truncnorm


from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from openfold.model.pair_transition import PairTransition

class LocalTriangleAttentionNew(nn.Module):
    def __init__(
            self,
            c_z=128, 
            c_hidden_mul=128, 
            pair_dropout=0.25, 
        ):
        super(LocalTriangleAttentionNew, self).__init__()
        
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z,c_hidden_mul,)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z,c_hidden_mul,)
        
        self.dropout_row_layer = DropoutRowwise(pair_dropout)
        self.dropout_col_layer = DropoutColumnwise(pair_dropout)
       
        self.layer_norm = nn.LayerNorm(c_z)

    def forward(self, edge_embed,  edge_mask):

        z = edge_embed
        z = z + self.dropout_row_layer(self.tri_mul_out(z, mask=edge_mask))
        z = z + self.dropout_row_layer(self.tri_mul_in(z, mask=edge_mask))
        return z

class IpaScore(nn.Module):

    def __init__(self, model_conf, edge_in=256, transition_n=4): 
        super(IpaScore, self).__init__()

        self.ipa_conf_num_blocks = 4
        self.ipa_conf_triangle_attention = True
        self.trunk = nn.ModuleDict()
        self.edge_in = edge_in
        self.transition_n = transition_n

        for b in range(self.ipa_conf_num_blocks): # 4
            if b < self.ipa_conf_num_blocks-1:

                self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew(c_z=self.edge_in, c_hidden_mul=self.edge_in)
            else:

                self.trunk[f'edge_transition_{b}'] = PairTransition(self.edge_in,self.transition_n,)
        
    def layer(self,cur_block, edge_embed,edge_mask,):
        if cur_block < self.ipa_conf_num_blocks-1:
            if self.ipa_conf_triangle_attention:
                edge_embed = self.trunk[f'edge_transition_{cur_block}'](
                    edge_embed, edge_mask=edge_mask)

        else:
            edge_embed = self.trunk[f'edge_transition_{cur_block}'](
                edge_embed, mask=edge_mask, chunk_size=4)
        edge_embed *= edge_mask[..., None]

        return edge_embed


    def forward(self, edge_embed,edge_mask=None):
      
        if edge_mask is None:
            edge_mask = torch.ones_like(edge_embed[..., 0], dtype=torch.float32, device=edge_embed.device)

        for b in range(self.ipa_conf_num_blocks):
            if b < self.ipa_conf_num_blocks-1:
                edge_embed = self.layer(
                    cur_block=b,
                    edge_embed=edge_embed,
                    edge_mask=edge_mask,)
        
                
            else:
                edge_embed = edge_embed + self.layer(cur_block=b, edge_embed=edge_embed,edge_mask=edge_mask)

        return edge_embed


if __name__ == '__main__':
    # # Example test
    B = 1    # batch size
    L = 10     # number of residues/nodes
    D = 128    # edge embedding dimension

    # Random edge features and edge mask
    edge_embed = torch.randn(B, L, L, D)  # shape: (B, L, L, D)
    edge_mask = torch.ones(B, L, L)  # 0 or 1

    # Instantiate and run
    model = IpaScore(model_conf=True)
    output = model(edge_embed, edge_mask)

    print("Output shape:", output.shape)
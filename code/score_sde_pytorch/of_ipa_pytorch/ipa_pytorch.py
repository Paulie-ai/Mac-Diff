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
            c_z=256, # 128
            c_hidden_mul=256, # 128
            pair_dropout=0.25, # 0.25
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

    def __init__(self, model_conf): 
        super(IpaScore, self).__init__()
        # self._model_conf = model_conf
        # ipa_conf = model_conf.ipa
        # self._ipa_conf = ipa_conf
        self.ipa_conf_num_blocks = 2
        self.ipa_conf_triangle_attention = True
        self.trunk = nn.ModuleDict()

        for b in range(self.ipa_conf_num_blocks): # 4
            if b < self.ipa_conf_num_blocks-1:
                # No edge update on the last block.
                # if self.ipa_conf_triangle_attention:
                    # use local edge triangle attention for better performance
                self.trunk[f'edge_transition_{b}'] = LocalTriangleAttentionNew()
            else:
                # use simple transition layer
                edge_in = 256
                # ipa_conf_c_s = 256
                transition_n = 4
                # self.trunk[f'edge_transition_{b}'] = EdgeTransition(
                #     node_embed_size=ipa_conf_c_s, # 256
                #     edge_embed_in=edge_in,
                #     edge_embed_out=edge_in,
                # )
                self.trunk[f'edge_transition_{b}'] = PairTransition(edge_in,transition_n,)
        # print(self.trunk)
    def layer(self,cur_block, edge_embed,edge_mask,):
        if cur_block < self.ipa_conf_num_blocks-1:
            if self.ipa_conf_triangle_attention:
                edge_embed = self.trunk[f'edge_transition_{cur_block}'](
                    edge_embed, edge_mask=edge_mask)
                # edge_embed *= edge_mask[..., None]
        else:
            edge_embed = self.trunk[f'edge_transition_{cur_block}'](
                edge_embed, mask=edge_mask, chunk_size=4)
        edge_embed *= edge_mask[..., None]

        return edge_embed


    def forward(self, edge_embed,edge_mask):
        # print('raw',edge_embed)

        for b in range(self.ipa_conf_num_blocks):
            if b < self.ipa_conf_num_blocks-1:
                edge_embed = self.layer(
                    cur_block=b,
                    edge_embed=edge_embed,
                    edge_mask=edge_mask,)
        
                # print(edge_embed[0])
            else:
                edge_embed = edge_embed + self.layer(cur_block=b, edge_embed=edge_embed,edge_mask=edge_mask)
        # print('transition',edge_embed)

        return edge_embed



if __name__ == '__main__':
    # # Example test
    B = 1    # batch size
    L = 10     # number of residues/nodes
    D = 256    # edge embedding dimension

    # Random edge features and edge mask
    edge_embed = torch.randn(B, L, L, D)  # shape: (B, L, L, D)
    edge_mask = torch.ones(B, L, L)  # 0 or 1

    # Instantiate and run
    model = IpaScore(model_conf=True)
    output = model(edge_embed, edge_mask)

    print("Output shape:", output.shape)




# def _prod(nums):
#     out = 1
#     for n in nums:
#         out = out * n
#     return out


# def _calculate_fan(linear_weight_shape, fan="fan_in"):
#     fan_out, fan_in = linear_weight_shape

#     if fan == "fan_in":
#         f = fan_in
#     elif fan == "fan_out":
#         f = fan_out
#     elif fan == "fan_avg":
#         f = (fan_in + fan_out) / 2
#     else:
#         raise ValueError("Invalid fan option")

#     return f

# def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
#     shape = weights.shape
#     f = _calculate_fan(shape, fan)
#     scale = scale / max(1, f)
#     a = -2
#     b = 2
#     std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
#     size = _prod(shape)
#     samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
#     samples = np.reshape(samples, shape)
#     with torch.no_grad():
#         weights.copy_(torch.tensor(samples, device=weights.device))


# def lecun_normal_init_(weights):
#     trunc_normal_init_(weights, scale=1.0)


# def he_normal_init_(weights):
#     trunc_normal_init_(weights, scale=2.0)


# def glorot_uniform_init_(weights):
#     nn.init.xavier_uniform_(weights, gain=1)


# def final_init_(weights):
#     with torch.no_grad():
#         weights.fill_(0.0)


# def gating_init_(weights):
#     with torch.no_grad():
#         weights.fill_(0.0)


# def normal_init_(weights):
#     torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")

# class Linear(nn.Linear):
#     """
#     A Linear layer with built-in nonstandard initializations. Called just
#     like torch.nn.Linear.

#     Implements the initializers in 1.11.4, plus some additional ones found
#     in the code.
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         bias: bool = True,
#         init: str = "default",
#         init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
#     ):
#         """
#         Args:
#             in_dim:
#                 The final dimension of inputs to the layer
#             out_dim:
#                 The final dimension of layer outputs
#             bias:
#                 Whether to learn an additive bias. True by default
#             init:
#                 The initializer to use. Choose from:

#                 "default": LeCun fan-in truncated normal initialization
#                 "relu": He initialization w/ truncated normal distribution
#                 "glorot": Fan-average Glorot uniform initialization
#                 "gating": Weights=0, Bias=1
#                 "normal": Normal initialization with std=1/sqrt(fan_in)
#                 "final": Weights=0, Bias=0

#                 Overridden by init_fn if the latter is not None.
#             init_fn:
#                 A custom initializer taking weight and bias as inputs.
#                 Overrides init if not None.
#         """
#         super(Linear, self).__init__(in_dim, out_dim, bias=bias)

#         if bias:
#             with torch.no_grad():
#                 self.bias.fill_(0)

#         if init_fn is not None:
#             init_fn(self.weight, self.bias)
#         else:
#             if init == "default":
#                 lecun_normal_init_(self.weight)
#             elif init == "relu":
#                 he_normal_init_(self.weight)
#             elif init == "glorot":
#                 glorot_uniform_init_(self.weight)
#             elif init == "gating":
#                 gating_init_(self.weight)
#                 if bias:
#                     with torch.no_grad():
#                         self.bias.fill_(1.0)
#             elif init == "normal":
#                 normal_init_(self.weight)
#             elif init == "final":
#                 final_init_(self.weight)
#             else:
#                 raise ValueError("Invalid init string.")


# class EdgeTransition(nn.Module):
#     def __init__(
#             self,
#             *,
#             node_embed_size,
#             edge_embed_in,
#             edge_embed_out,
#             num_layers=2,
#             node_dilation=2
#         ):
#         super(EdgeTransition, self).__init__()

#         bias_embed_size = node_embed_size // node_dilation
#         self.initial_embed = Linear(
#             node_embed_size, bias_embed_size, init="relu")
#         hidden_size = bias_embed_size * 2 + edge_embed_in
#         trunk_layers = []
#         for _ in range(num_layers):
#             trunk_layers.append(Linear(hidden_size, hidden_size, init="relu"))
#             trunk_layers.append(nn.ReLU())
#         self.trunk = nn.Sequential(*trunk_layers)
#         self.final_layer = Linear(hidden_size, edge_embed_out, init="final")
#         self.layer_norm = nn.LayerNorm(edge_embed_out)

#     def forward(self, node_embed, edge_embed):
#         node_embed = self.initial_embed(node_embed)
#         batch_size, num_res, _ = node_embed.shape
#         edge_bias = torch.cat([
#             torch.tile(node_embed[:, :, None, :], (1, 1, num_res, 1)),
#             torch.tile(node_embed[:, None, :, :], (1, num_res, 1, 1)),
#         ], axis=-1)
#         edge_embed = torch.cat(
#             [edge_embed, edge_bias], axis=-1).reshape(
#                 batch_size * num_res**2, -1)
#         edge_embed = self.final_layer(self.trunk(edge_embed) + edge_embed)
#         edge_embed = self.layer_norm(edge_embed)
#         edge_embed = edge_embed.reshape(
#             batch_size, num_res, num_res, -1
#         )
#         return edge_embed



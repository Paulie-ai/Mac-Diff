# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

# Adapted from score_sde_pytorch: https://github.com/yang-song/score_sde_pytorch
# Removed progressive module and Fourier timestep embeddings and FIR kernel

from tkinter import NO

from numpy import isin
from . import layers, normalization, utils, attention, ipa_pytorch
import torch.nn as nn
import functools
import torch
from abc import abstractmethod


ResnetBlockDDPM = layers.ResnetBlockDDPMpp
ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
# Combine = layers.Combine
conv3x3 = layers.conv3x3
conv1x1 = layers.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
SpatialTransformer = attention.SpatialTransformer

def distance_to_gaussian(distance_matrix, sigma=16.0):
    """
    Convert a distance matrix to a Gaussian probability matrix.
    """
    N = distance_matrix.size(0)
    gaussian_probs = torch.exp(-distance_matrix**2 / (2 * sigma**2))
    return gaussian_probs

class TimestepBlock(nn.Module):
  """
  Any module where forward() takes timestep embeddings as a second argument.
  """

  @abstractmethod
  def forward(self, x, emb):
    """
    Apply the module to `x` given `emb` timestep embeddings.
    """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, context=None, context_emb=None, img_masks=None):
        for layer in self:

          if type(layer) in (layers.ResnetBlockDDPMpp, layers.ResnetBlockBigGANpp):
              x = layer(x, emb)
          elif isinstance(layer, SpatialTransformer):
              x = layer(x, context, context_emb)
          elif isinstance(layer, layers.AttnBlockpp):
              x = layer(x)
          else:
              x = layer(x)

        return x


class UNetModel(nn.Module):
  """UNet model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config), dtype=torch.float32))
    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.init_self = config.model.init_self
    self.weight_scale = config.model.weight_scale
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    self.dp = config.model.embedding_dropout
    self.use_flash_attn = config.model.use_flash_attn
    resamp_with_conv = config.model.resamp_with_conv
    self.gaus_sigma = config.model.gaus_sigma
    self.gaus_weight = config.model.gaus_weight
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.max_res_num // (2 ** i) for i in range(num_resolutions)]
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    init_scale = config.model.init_scale
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    self.n_heads = n_heads = config.model.n_heads
    self.context_dim = context_dim = self.config.model.context_dim
    self.q = nn.Linear(self.context_dim, self.context_dim, bias=False)
    self.k = nn.Linear(self.context_dim, self.context_dim, bias=False)
    self.v = nn.Linear(self.context_dim, self.context_dim, bias=False)
    self.attn_out = nn.Sequential(
      nn.LayerNorm(self.context_dim),
      nn.Linear(self.context_dim, self.context_dim*4),
      nn.GELU(),
      nn.Linear(self.context_dim*4, self.context_dim)
    )

    for layer in [self.q, self.k, self.v]:
      for p in layer.parameters():
        p.detach().copy_(torch.eye(*p.shape))

    nn.init.zeros_(self.attn_out[-1].weight)
    nn.init.zeros_(self.attn_out[-1].bias)

    self.alpha = nn.Parameter(torch.tensor(config.model.raw_weight).cuda())
    assert embedding_type in ['fourier', 'positional']

    modules = []
    embed_dim = nf
    modules.append(nn.Linear(embed_dim, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    self.pre_blocks = nn.ModuleList(modules)

    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale,
                                  use_flash_attn=self.use_flash_attn)

    Upsample = functools.partial(layers.Upsample,
                                 with_conv=resamp_with_conv)

    Downsample = functools.partial(layers.Downsample,
                                   with_conv=resamp_with_conv)

    # using biggan
    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')
    # 5
    channels = config.data.num_channels
    self.pre_conv = conv3x3(channels, nf)

    # Downsampling block
    self.input_blocks = nn.ModuleList([])
    hs_c = [nf]
    in_ch = nf
    self.input_channels = [nf]

    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks): 
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch, out_ch=out_ch)]

        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                               n_heads=n_heads,
                                               d_head=in_ch // n_heads,
                                               context_dim=256))
        self.input_blocks.append(TimestepEmbedSequential(*cur_layers))
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

      if i_level != num_resolutions - 1:
        self.input_blocks.append(
          TimestepEmbedSequential(
            Downsample(in_ch=in_ch) if resblock_type == 'ddpm'
            else ResnetBlock(down=True, in_ch=in_ch)
          )
        )
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

    in_ch = hs_c[-1]
    self.mid_channel = self.input_channels[-1]
    self.mid_blocks = TimestepEmbedSequential(
      ResnetBlock(in_ch=self.mid_channel),
      AttnBlock(channels=self.mid_channel),
      SpatialTransformer(in_channels=in_ch,
                         n_heads=n_heads,
                         d_head=in_ch // n_heads,
                         context_dim=256),
      ResnetBlock(in_ch=self.mid_channel)
    )

    # Upsampling block
    self.out_blocks = nn.ModuleList([])
    self.out_channels = []
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch + self.input_channels.pop(), out_ch=out_ch)]
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                               n_heads=n_heads,
                                               d_head=in_ch // n_heads,
                                               context_dim=256)) # context_dim

        if i_level != 0 and i_block == num_res_blocks:
          if resblock_type == 'ddpm':
            cur_layers.append(Upsample(in_ch=in_ch))
          else:
            cur_layers.append(ResnetBlock(in_ch=in_ch, up=True))
        self.out_blocks.append(TimestepEmbedSequential(*cur_layers))

    self.out = nn.ModuleList([])
    self.out.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    self.out.append(self.act)
    self.out.append(conv3x3(in_ch, channels, init_scale=init_scale))
    
    self.ipascore = ipa_pytorch.IpaScore(model_conf=True, edge_in=512)
    
    self.o_proj = nn.Linear(config.model.context_dim, 512, bias=True)
    torch.nn.init.zeros_(self.o_proj.bias)
    self.layernorm = nn.LayerNorm(self.context_dim*2)

  def forward(self, x, time_cond, text_emb=None, esm_contact=None): # 
  
    batch_size, shapes = text_emb.shape[0], text_emb.shape[1]
    p = torch.arange(0, text_emb.shape[1], device=torch.device('cuda'))
    p = torch.abs(p[None,:] - p[:,None])
    gaussian_probs = distance_to_gaussian(p, self.gaus_sigma)
    bs_gaussian_prbs = gaussian_probs.unsqueeze(0).expand(text_emb.shape[0],-1,-1)
    diagonal_masks = torch.where(esm_contact == 0, torch.tensor(0, dtype=torch.float, device=torch.device('cuda')), torch.tensor(1, dtype=torch.float, device=torch.device('cuda')))
    bs_gaussian_prbs = bs_gaussian_prbs.masked_fill(~diagonal_masks.bool(), 0.)
    q, k, v = self.q(text_emb), self.k(text_emb), self.v(text_emb)
    att_weights = torch.einsum('bid,bjd -> bij', q, k) * (k.shape[-1] ** -0.5)
    att_weights = att_weights.masked_fill(~diagonal_masks.bool(), -1e9)
    att_weights = nn.functional.softmax(att_weights, dim=-1) * diagonal_masks  # [0, 1]
    weight =  (1-self.gaus_weight) * esm_contact + self.gaus_weight * bs_gaussian_prbs
    weight = weight / (weight.sum(-1, keepdim=True) + torch.full_like(weight, 1e-9, device=torch.device('cuda')))
    self.weight_scale = torch.sigmoid(self.alpha)
    weight = self.weight_scale * weight + (1-self.weight_scale) * att_weights
    text_emb = torch.matmul(weight, v)
    
    text_emb = text_emb + self.attn_out(text_emb)

    text_concat = torch.cat([text_emb.unsqueeze(2).expand(-1,-1,shapes,-1), text_emb.unsqueeze(1).expand(-1,shapes,-1,-1)],dim=-1)
    text_concat = self.layernorm(text_concat)
    text_concat = self.o_proj(text_concat)
    residue_short = text_concat
    text_concat = self.ipascore(text_concat, edge_mask=diagonal_masks)
    text_concat += residue_short
    
    timesteps = time_cond
    used_sigmas = self.sigmas[time_cond.long()]
    temb = layers.get_timestep_embedding(timesteps, self.nf) 
    
    # pre blocks
    for module in self.pre_blocks:
      temb = module(temb)

    x = x.float()
    h = self.pre_conv(x)
    hs = [h]

    # Down Sampling blocks
    for i, module in enumerate(self.input_blocks):
      h = module(h, temb, text_concat, text_emb)
      hs.append(h)

    # Mid Block
    h = self.mid_blocks(h, temb, text_concat, text_emb)

    # Up Sampling blocks
    for module in self.out_blocks:
        h = torch.cat([h, hs.pop()], dim=1) 
        h = module(h, temb, text_concat, text_emb)

    assert not hs

    for out in self.out:
      h = out(h)

    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h
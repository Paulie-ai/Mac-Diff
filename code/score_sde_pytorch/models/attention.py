from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def naive_downsample_2d(x, factor=2):
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, att_dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.att_dropout = att_dropout

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.fc = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                nn.ReLU(),
                                nn.Linear(emb_dim, emb_dim))

    def forward(self, x, pad_mask=None):
        # [batch_szie, seq_len, emb_dim]
        batch_size = x.size(0)

        # [batch_szie, seq_len, emb_dim]
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights / math.sqrt(self.depth)

        if pad_mask is not None:
            # [batch_size, seq_len, seq_len] -> [batch_size, nums_head, seq_len, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill((~pad_mask.bool()), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)

    
        if self.att_dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.att_dropout)

        output = torch.matmul(att_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        output = self.fc(output)

        return output

class Cross_MultiAttention(nn.Module):
    def __init__(self, num_heads, in_channel, emb_dim, d_context, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads
        self.scale = self.depth ** -0.5

        self.Wq = nn.Linear(in_channel, emb_dim)
        self.Wk = nn.Linear(d_context, emb_dim)
        self.Wv = nn.Linear(d_context, emb_dim)

        self.proj_out = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                nn.ReLU(),
                                nn.Linear(emb_dim, in_channel))


    def forward(self, x, context, pad_mask=None, seq_mask=None):
        '''
        :param x: [batch_size, h*w, c]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len*seq_len]
        :param seq_mask: [batch_size, seq_len]
        :return:
        '''
        batch_size, h, c = x.shape

        Q = self.Wq(x)
        K = self.Wk(context)
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            pad_mask = torch.matmul(pad_mask[..., None], seq_mask[:, None, :])
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(~(pad_mask.bool()), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)

        out = self.proj_out(out)

        return out
    
class Hard_or_Soft_Attention(nn.Module):
    def __init__(self, in_channel=2560+128, out_channel=128):
        super(Hard_or_Soft_Attention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_channel, out_channel*4),
                                nn.GELU(),
                                nn.Linear(out_channel*4, out_channel)) 

    def forward(self, x, context, seqs_embedding=None):
        """
        concat contextual information from (x_i and x_j) as spatial alignment.
        """
        n, h_w, c = x.shape
        context_concat = context.view((n, h_w, -1))
        x = torch.cat([x, context_concat], dim=-1)
        x = self.project(x)
        return x

def distance_to_gaussian(distance_matrix, sigma=16.0):
    """
    Convert a distance matrix to a Gaussian probability matrix.

    Parameters:
        distance_matrix (torch.Tensor): Input distance matrix of shape [N, N].
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Gaussian probability matrix of shape [N, N].
    """
    N = distance_matrix.size(0)
    gaussian_probs = torch.exp(-distance_matrix**2 / (2 * sigma**2))

    return gaussian_probs

'''
Reference: https://github.com/hkproj/pytorch-stable-diffusion/blob/e0cb06de011787cdf13eed7b4287ad8410491149/sd/diffusion.py#L78
'''
class SpatialTransformer(nn.Module):
    #  in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None
    def __init__(self, in_channels, n_heads: int, d_head: int, depth=1, dropout=0., context_dim=1280):
        super().__init__() 
        channels = n_heads * d_head
        assert in_channels == channels

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = MultiHeadAttention(n_heads, channels)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = Hard_or_Soft_Attention(in_channel=context_dim*2+channels, out_channel=channels)  
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context, seq_emb=None, contact_map=None, mask=None):

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        assert h==w
        seq_shape = context.shape[2]
        cycle = int(math.log2(seq_shape/h))
        context = context.permute(0, 3, 1, 2)
        for t in range(cycle):
            context = naive_downsample_2d(context)
        context = context.permute(0, 2, 3, 1) 
        assert h == context.shape[1]
        
        x = x.view((n, c, h * w))   # (n, c, hw)
        x = x.transpose(-1, -2)  # (n, hw, c)

        residue_short = x
        x = self.layernorm_2(x)  # [n, hw, c]
        x = self.attention_2(x, context, seq_emb)
        x += residue_short 

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2).contiguous()
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long
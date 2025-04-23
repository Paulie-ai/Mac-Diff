# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_wwk.utils_torch import Bottle2neck
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint

from .modules_wwk import (
    TransformerLayer,
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
    OuterProductSeq,
    OuterProductMSA
)

from .axial_attention import RowSelfAttention, ColumnSelfAttention


class ProteinBertModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--layers", default=33, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument(
            "--arch",
            default='roberta_large',
            type=str,

        )

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        if self.args.arch == 'roberta_large':
            self.model_version = 'ESM-1b'
            self._init_submodules_esm1b()
        else:
            self.model_version = 'ESM-1'
            self._init_submodules_esm1()

        self.layers_res2net = nn.ModuleList([])
        self._init_submodules_res2net(Bottle2neck, 64, 28, expansion=2)
        self._init_submodules_res2net(Bottle2neck, 128, 28, expansion=2)

    def _init_submodules_common(self):
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )
        dim_pair = [None] * 5 + [128] * 14 + [256] * 14
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim, self.args.ffn_embed_dim, self.args.attention_heads,
                    add_bias_kv=(self.model_version != 'ESM-1b'),
                    use_esm1b_layer_norm=(self.model_version == 'ESM-1b'),
                    dim_pair=dim_pair[i]
                )
                for i in range(self.args.layers)
            ]
        )
        dim_pair[:6] = [64] * 6
        dim_pair[5 + 14] = 128
        self.outer_product = nn.ModuleList(
            [
                OuterProductSeq(out_dim=dim_pair[i])
                for i in range(self.args.layers)
            ]
        )
        self.linear_attn = nn.ModuleList(
            [
                nn.Conv2d(kernel_size=1, in_channels=20, out_channels=dim_pair[i], bias=False)
                for i in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def _init_submodules_esm1b(self):
        self._init_submodules_common()
        self.embed_scale = 1
        self.embed_positions = LearnedPositionalEmbedding(self.args.max_positions, self.args.embed_dim, self.padding_idx)
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight
        )

    def _init_submodules_esm1(self):
        self._init_submodules_common()
        self.embed_scale = math.sqrt(self.args.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(self.args.embed_dim, self.padding_idx)
        self.embed_out = nn.Parameter(
            torch.zeros((self.alphabet_size, self.args.embed_dim))
        )
        self.embed_out_bias = None
        if self.args.final_bias:
            self.embed_out_bias = nn.Parameter(torch.zeros(self.alphabet_size))

    def _init_submodules_res2net(self, block, planes, blocks, stride=1, expansion=4, baseWidth=26, scale=4):
        self.layers_res2net.append(block(planes, planes, stride, expansion=expansion, stype='stage', baseWidth=baseWidth, scale=scale))
        inplanes = planes * expansion
        d = 1
        for i in range(1, blocks):
            d = 2 * d % 31
            self.layers_res2net.append(block(inplanes, planes, expansion=expansion, baseWidth=baseWidth, scale=scale, dilation=d))

    def forward(self, tokens, f2d, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if getattr(self.args, 'token_dropout', False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens)

        if self.model_version == 'ESM-1b':
            x = self.emb_layer_norm_before(x)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            try:
                cuda_lst = self.args.cuda_lst
            except AttributeError:
                cuda_lst = None
            if cuda_lst is not None and len(cuda_lst) > 1:
                if layer_idx >= len(self.layers) // len(cuda_lst):
                    layer = layer.cuda(2)
                    if layer_idx == len(self.layers) // len(cuda_lst):
                        x = x.cuda(2)

            if layer_idx >= 5:
                f2d = f2d + Rearrange('b i j d->b d i j')(self.outer_product[layer_idx](x))
                f2d = f2d + self.linear_attn[layer_idx](Rearrange('h b i j->b h i j')(attn))
                f2d = self.layers_res2net[2 * (layer_idx - 5)](f2d)
                f2d = self.layers_res2net[2 * (layer_idx - 5) + 1](f2d)
                x, attn = layer(x, pair_bias=f2d, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights)
            else:
                x, attn = layer(x, pair_bias=None, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights)

            if (layer_idx + 1) in repr_layers:
                if cuda_lst is not None and len(cuda_lst) > 1:
                    hidden_representations[layer_idx + 1] = x.transpose(0, 1).cuda(cuda_lst[0])
                else:
                    hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                if cuda_lst is not None and len(cuda_lst) > 1:
                    attn_weights.append(attn.transpose(1, 0).cuda(cuda_lst[0]))
                else:
                    attn_weights.append(attn.transpose(1, 0))
            if (cuda_lst is not None and len(cuda_lst) > 1) and layer_idx == len(self.layers) - 1:
                x = x.cuda(1)

        if self.model_version == 'ESM-1b':
            x = self.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            x = self.lm_head(x)
        else:
            x = F.linear(x, self.embed_out, bias=self.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations, 'logits2d': f2d}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = (1 - padding_mask.type_as(attentions))
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers


class MSATransformer(nn.Module):

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=12, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=768, type=int, metavar="N", help="embedding dimension"
        )
        # parser.add_argument(
        #     "--logit_bias", action="store_true", help="whether to apply bias to logits"
        # )
        parser.add_argument(
            "--ffn_embed_dim",
            default=3072,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=12,
            type=int,
            metavar="N",
            help="number of attention heads",
        )
        parser.add_argument(
            "--dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--attention_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--activation_dropout",
            default=0.1,
            type=float,
            help="Dropout to apply."
        )
        parser.add_argument(
            "--max_tokens_per_msa",
            default=2 ** 14,
            type=int,
            help=(
                "Used during inference to batch attention computations in a single "
                "forward pass. This allows increased input sizes with less memory."
            ),
        )

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
        )

        if getattr(args, "embed_positions_msa", False):
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, 1),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)

        self.dropout_module = nn.Dropout(self.args.dropout)
        dim_pair = [64] + [128] * 6 + [256] * 5
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    self.args.dropout,
                    self.args.attention_dropout,
                    self.args.activation_dropout,
                    getattr(self.args, "max_tokens_per_msa", self.args.max_tokens),
                    dim_pair=dim_pair[i]
                )
                for i in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.embed_positions = LearnedPositionalEmbedding(
            self.args.max_positions, self.args.embed_dim, self.padding_idx,
        )
        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.args.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight
        )
        self.layers_res2net = nn.ModuleList([])
        self._init_submodules_res2net(Bottle2neck, 64, 5, 6, expansion=2)
        self._init_submodules_res2net(Bottle2neck, 128, 5, 6, expansion=2)

        self.msa2pair = nn.ModuleList([
            OuterProductMSA(out_dim=dim_pair[i])
            for i in range(self.args.layers)
        ])

    def _init_submodules_res2net(self, block, planes, n_iters, groups=6, stride=1, expansion=4, baseWidth=26, scale=4):
        for g in range(groups):
            blocks = []
            if g == 0:
                blocks.append(block(planes, planes, stride, expansion=expansion, stype='stage', baseWidth=baseWidth, scale=scale))
                inplanes = planes * expansion
            else:
                blocks.append(block(inplanes, planes, stride, expansion=expansion, baseWidth=baseWidth, scale=scale))
            d = 1
            for i in range(1, n_iters):
                d = 2 * d % 31
                blocks.append(block(inplanes, planes, expansion=expansion, baseWidth=baseWidth, scale=scale, dilation=d))
            self.layers_res2net.append(nn.Sequential(*blocks))

    def forward(
            self,
            tokens,
            f2d,
            repr_layers=[],
            need_head_weights=False,
            return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(tokens)
        x += self.embed_positions(
            tokens.view(batch_size * num_alignments, seqlen)
        ).view(x.size())
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attn_weights = []
            col_attn_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                pair_bias=f2d,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                x, row_attn = x
                # H x C x B x R x R -> B x H x C x R x R
                # col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)
            f2d = f2d + self.msa2pair[layer_idx](x)
            f2d = Rearrange('b i j d->b d i j')(f2d)
            f2d = self.layers_res2net[layer_idx](f2d)
            f2d = Rearrange('b d i j->b i j d')(f2d)

        # x = self.emb_layer_norm_after(x)
        # x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        #
        # # last hidden representation should have layer norm applied
        # if (layer_idx + 1) in repr_layers:
        #     hidden_representations[layer_idx + 1] = x
        # x = self.lm_head(x)

        # result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # col_attentions: B x L x H x C x R x R
            # col_attentions = torch.stack(col_attn_weights, 1)
            # row_attentions: B x L x H x C x C
            row_attentions = torch.stack(row_attn_weights, 1)
            # result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_head(
                    tokens, row_attentions
                )
                result["contacts"] = contacts

        return f2d

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers

    def max_tokens_per_msa_(self, value: int) -> None:
        """ The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value

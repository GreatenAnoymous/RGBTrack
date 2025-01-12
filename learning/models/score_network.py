# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from network_modules import *
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``forward()`` will use a special optimized implementation if all of the following
    conditions are met:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor. This
      restriction will be loosened in the future.)
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - dropout is 0
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - at most one of ``key_padding_mask`` or ``attn_mask`` is passed
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed

    If the optimized implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True, attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3
        
        if self.batch_first and is_batched:
            # Transpose batch dimension to second position
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        # Get sizes
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        scaling = float(self.head_dim) ** -0.5
        
        # Linear projections for Q, K, V
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                # Self-attention
                q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            else:
                # Cross-attention
                w_q, w_k, w_v = self.in_proj_weight.chunk(3)
                b_q, b_k, b_v = None, None, None
                if self.in_proj_bias is not None:
                    b_q, b_k, b_v = self.in_proj_bias.chunk(3)
                q = F.linear(query, w_q, b_q)
                k = F.linear(key, w_k, b_k)
                v = F.linear(value, w_v, b_v)
        else:
            q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:self.embed_dim] if self.in_proj_bias is not None else None)
            k = F.linear(key, self.k_proj_weight, self.in_proj_bias[self.embed_dim:2*self.embed_dim] if self.in_proj_bias is not None else None)
            v = F.linear(value, self.v_proj_weight, self.in_proj_bias[2*self.embed_dim:] if self.in_proj_bias is not None else None)
        
        # Reshape Q, K, V for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Calculate attention scores
        attn_output_weights = torch.bmm(q * scaling, k.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if key_padding_mask.dtype == torch.bool:
                attn_output_weights = attn_output_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            else:
                attn_output_weights = attn_output_weights + key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Apply softmax
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        
        # Apply dropout
        if self.dropout > 0:
            attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        
        # Calculate attention output
        attn_output = torch.bmm(attn_output_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        
        # Handle attention weights for return
        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
        else:
            attn_output_weights = None
        
        # Restore batch dimension to first position if needed
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)
        
        return attn_output, attn_output_weights
class ScoreNetMultiPair(nn.Module):
    def __init__(self, cfg=None, c_in=6):
        super().__init__()
        self.cfg = cfg
        if self.cfg is None or self.cfg.use_BN:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None

        self.encoderA = nn.Sequential(
        ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
        ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
        )

        self.encoderAB = nn.Sequential(
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
        ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4
        # self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
        self.att= MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
        self.att_cross = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
        # self.att_cross = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)

        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
        self.linear = nn.Linear(embed_dim, 1)


    def extract_feat(self, A, B):
        """
        @A: (B*L,C,H,W) L is num of pairs
        """
        bs = A.shape[0]  # B*L

        x = torch.cat([A,B], dim=0)
        x = self.encoderA(x)
        a = x[:bs]
        b = x[bs:]
        ab = torch.cat((a,b), dim=1)
        ab = self.encoderAB(ab)
        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
        ab, _ = self.att(ab, ab, ab)
        return ab.mean(dim=1).reshape(bs,-1)


    def forward(self, A, B):
        """
        @A: (B*L,C,H,W) L is num of pairs
        @L: num of pairs
        """
        output = {}
        L= A.shape[0]
        bs = A.shape[0]//L
        feats = self.extract_feat(A, B)   #(B*L, C)
        x = feats.reshape(bs,L,-1)
        x, _ = self.att_cross(x, x, x)

        output['score_logit'] = self.linear(x).reshape(bs,L)  # (B,L)

        return output

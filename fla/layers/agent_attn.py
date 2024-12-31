# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import RMSNorm
from fla.modules.feature_map import (DPFPFeatureMap, HadamardFeatureMap,
                                     HedgehogFeatureMap, T2RFeatureMap)
from fla.ops.agent_attention import naive_chunk_agent_attn


class AgentAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'naive_agent_attention',
        hidden_size: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        # standard linear attention normalization
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        agent_num=4,
        sequence_length = 16,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.agent_num = agent_num
        self.sequence_length = sequence_length
        self.pool = nn.Conv2d(in_channels= self.sequence_length, out_channels=self.agent_num, kernel_size=1)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert mode in ['naive_agent_attention'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)

        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, x):
        mode = self.mode
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # [batch, sequence, head, dimension]

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        agent_tokens = self.pool(q)
        # [batch, agent_dimension, head, dimension]
        if self.num_kv_groups > 1:
            k, v = (repeat(x, '... (h d) -> ... (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, '... (h d) -> ... h d', h=self.num_kv_heads) for x in (k, v))

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)

        if mode == 'naive_agent_attention':
            o = naive_chunk_agent_attn(
                q=q,
                k=k,
                v=v,
                agent_tokens = agent_tokens,
                normalize=self.do_feature_map_norm
            )
        else:
            raise NotImplementedError
        o = self.norm(o)
        batch_size, sequence_size, _, _ = o.shape
        o = o.transpose(1, 2).contiguous().view(batch_size, sequence_size, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        o = self.o_proj(o)
        return o

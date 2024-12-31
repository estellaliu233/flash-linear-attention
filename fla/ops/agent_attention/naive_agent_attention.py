# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from fla.ops.agent_attention.utils import normalize_output


def naive_chunk_agent_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    agent_tokens: torch.Tensor,
    scale: Optional[float] = None,
    normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 4
    q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
    agent_tokens = rearrange(agent_tokens, 'b (n c) h d -> b h n c d', c=chunk_size)
    softmax = nn.Softmax(dim=-1)
    agent_attn = softmax((agent_tokens * scale) @ k.transpose(-2, -1))
    q_attn = softmax(q @ agent_tokens.transpose(-2, -1))
    kv = agent_attn.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q_attn  @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
    if normalize:
        o = normalize_output(q * scale, k, o)
    return rearrange(o, 'b h n c d -> b (n c) h d')
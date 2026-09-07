"""Time/noise embeddings and the FlowFormer used by the structure network."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .trRosettaX2.evoutils.attn_conv_e2e import TriUpdate, UpdateM, UpdateX


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for scalar noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embedding_size // 2) * scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


class Derf(nn.Module):
    """Differential-erf normalizing unit; drops in wherever LayerNorm is used."""

    def __init__(self, normalized_shape, eps=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.s = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta

    def extra_repr(self):
        return '{normalized_shape}'.format(**self.__dict__)


class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, pair, m):
        for tri_update, update_x, update_m in self.blocks:
            x = update_x(x, m)        # MSA -> pair update
            x = tri_update(x)         # triangular pair update
            x = x + pair              # residual from the semantic pair repr
            m_out = update_m(x, m)    # pair -> MSA update
            m = m_out + m
        return x, m


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class FlowFormer(nn.Module):
    """Fuse noisy structural, semantic pair, and MSA representations.

    The ``net.blocks`` module hierarchy intentionally matches the released
    checkpoints. The public architecture terminology can therefore use
    ``FlowFormer`` without invalidating trained weights.
    """

    def __init__(self, dim, no_blocks, use_derf=False):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(no_blocks):
            layers.append(nn.ModuleList([
                TriUpdate(in_dim=dim, use_derf=use_derf),
                UpdateX(in_dim=dim, dim=dim),
                UpdateM(in_dim=dim, pair_dim=dim, use_derf=use_derf),
            ]))
        self.net = SequentialSequence(layers)

    def forward(self, x, m, pair):
        x, m = self.net(x, pair, m)
        return x, m

"""Triangular-update building blocks used by the 2D trunk."""
import torch
import torch.nn as nn
from einops import rearrange


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


class TriangleMultiplication(nn.Module):
    def __init__(self, in_dim=128, dim=128, direct='outgoing', use_derf=False):
        super(TriangleMultiplication, self).__init__()
        self.direct = direct
        if use_derf:
            self.norm = Derf(in_dim)
        else:
            self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, dim * 2)
        self.linear2 = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.Sigmoid()
        )
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim, in_dim)
        if use_derf:
            self.to_out = nn.Sequential(
                Derf(dim),
                self.linear_out
            )
        else:
            self.to_out = nn.Sequential(
                nn.LayerNorm(dim),
                self.linear_out
            )

    def forward(self, z):
        direct = self.direct
        z = self.norm(z)
        a, b = torch.chunk(self.linear2(z) * self.linear1(z), 2, -1)
        gate = self.to_gate(z)
        if direct == 'outgoing':
            prod = torch.einsum('bikd,bjkd->bijd', a, b)
        elif direct == 'incoming':
            prod = torch.einsum('bkid,bkjd->bijd', a, b)
        else:
            raise ValueError('direct should be outgoing or incoming!')
        out = gate * self.to_out(prod)
        return out


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row', use_derf=False):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        if use_derf:
            self.norm = Derf(in_dim)
        else:
            self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z):
        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** .5
        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2
        else:
            raise ValueError('wise should be col or row!')
        attn = (torch.einsum(eq_attn, q, k) / scale + b).softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')
        z_ = self.to_out(out)
        return z_


class PairTransition(nn.Module):
    def __init__(self, dim=128, n=4, use_derf=False):
        super(PairTransition, self).__init__()
        if use_derf:
            self.norm = Derf(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * n)
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim * n, dim)
        )

    def forward(self, z):
        z = self.norm(z)
        a = self.linear1(z)
        z = self.linear2(a)
        return z

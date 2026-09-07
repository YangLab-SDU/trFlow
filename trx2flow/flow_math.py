"""Flow-matching math: rigid alignment, pseudo-CB extraction, harmonic prior."""
from __future__ import annotations

import numpy as np
import torch


def rmsdalign(a, b, weights=None):  # aligns B to A  # [*, N, 3]
    """Kabsch-style rigid alignment of ``b`` onto ``a`` (see scipy Rotation)."""
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[..., -1] *= sgn
    u[..., :, -1] *= sgn.unsqueeze(-1)
    C = u @ vh  # C rotates B to A
    return b @ C.mT + a_mean


atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

restypes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
restype_order = {restype: i for i, restype in enumerate(restypes)}


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Extract pseudo-CB coordinates (CA for glycine) from full-atom positions."""
    is_gly = aatype == restype_order["G"]
    ca_idx = atom_order["CA"]
    cb_idx = atom_order["CB"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx],
            all_atom_masks[..., cb_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


class HarmonicPrior:
    """Harmonic (spring-chain) prior over pseudo-CB chains; ``sample`` returns noise."""

    def __init__(self, N=256, a=3 / (3.8 ** 2)):
        J = torch.zeros(N, N)
        for i, j in zip(np.arange(N - 1), np.arange(1, N)):
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a
        D, P = torch.linalg.eigh(J)
        D_inv = 1 / D
        D_inv[0] = 0
        self.P, self.D_inv = P, D_inv
        self.N = N

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)

    def sample(self, batch_dims=()):
        return self.P @ (torch.sqrt(self.D_inv)[:, None] * torch.randn(*batch_dims, self.N, 3, device=self.P.device))

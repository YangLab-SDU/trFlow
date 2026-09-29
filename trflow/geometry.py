"""Structure geometry helpers: PDB parsing, distograms, bin conversions."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from Bio import PDB
from scipy.ndimage import gaussian_filter1d

res_name_dict = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "PHD": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "UNK": "X", "TYR": "Y", "VAL": "V", "SEC": "U",
    "ASX": "B", "GLX": "Z", "XLE": "J", "XAA": "X",
}
atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
]
retain_all_res = False

restypes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
restype_order = {restype: i for i, restype in enumerate(restypes)}
aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
aa_to_index = {aa: idx for idx, aa in enumerate(aa_list)}


def get_atom_positions_pdb(pdb_file, model=0, retain_all_res=True):
    """Parse a PDB file into per-atom xyz tensors ``{atom: [L, 3]}`` plus seq/res ids."""
    pp = PDB.PDBParser(QUIET=True)
    structure = pp.get_structure("", pdb_file)[model]
    xyzs_all = defaultdict(list)
    for chain in structure.child_list:
        residues = [res for res in chain.child_list if not res.id[0].strip()]
        seq_pdb = "".join(res_name_dict[res.resname.strip()] for res in residues)
        if retain_all_res:
            L = residues[-1].id[1]
            res_id = np.arange(L)
        else:
            L = len(residues)
            res_id = np.array([int(res.id[1]) for res in residues])

        xyzs = {}
        for atom in atom_types:
            xyzs[atom] = np.nan * np.zeros((L, 3))

        for i, res in enumerate(residues):
            for atom in atom_types:
                try:
                    coord = res[atom].coord
                    if retain_all_res:
                        xyzs[atom][res.id[1] - 1] = coord
                    else:
                        xyzs[atom][i] = coord
                except KeyError:
                    continue
        for atom in xyzs:
            xyzs_all[atom].append(xyzs[atom])

    for atom in xyzs_all:
        xyzs_all[atom] = np.concatenate(xyzs_all[atom], axis=0)
        xyzs_all[atom] = torch.from_numpy(xyzs_all[atom]).float()
    return xyzs_all, res_id, seq_pdb


def pseudo_beta_fn_from_pdb(pdb_file, all_atom_masks):
    """Extract pseudo-CB coordinates (CA for glycine) directly from a PDB file."""
    xyzs_all, _, seq_pdb = get_atom_positions_pdb(pdb_file)

    aatype = np.zeros((len(seq_pdb), 21))
    for i, aa in enumerate(seq_pdb):
        aatype[i, aa_to_index.get(aa, 20)] = 1
    aatype = torch.from_numpy(np.argmax(aatype, axis=-1))

    is_gly = aatype == restype_order["G"]
    pseudo_beta = torch.where(
        is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
        xyzs_all['CA'],
        xyzs_all['CB'],
    )
    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            xyzs_all['CA'],
            xyzs_all['CB'],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def params(flag):
    if flag == "0HHD":
        return 0, 0, 0.3, 0.03, 0.72
    if flag == "0LD":
        return 0, 0, 0.5, 0.07, 0.50
    if flag == "0HD":
        return 0, 0, 0.5, 0.05, 0.50
    if flag == "0LLD":
        return 0, 0, 0.7, 0.1, 0.42


def process_distribution_fast(unprocessed_dist, fact_dist, norm=True, smooth=True, sigma=1.0):
    """Blend a structural distance distribution into a predicted distogram.

    For entries whose predicted probability peak is low, the observed distance
    distribution ``fact_dist`` replaces the prediction with a decayed copy of
    itself. Returns the updated distribution (normalized when ``norm=True``).
    """
    tmp = np.copy(unprocessed_dist)
    backward, forward, P, pcut, decay_rate = params("0HD")

    mask = unprocessed_dist.max(axis=-1) < P

    tmp_sub = tmp[mask]
    fact_sub = fact_dist[mask]

    if tmp_sub.shape[0] > 0:
        size = tmp.shape[-1]
        idx = np.argmax(fact_sub, axis=-1)

        bw_arr = np.where(idx - backward >= 0, backward, idx)
        fw_arr = np.where(idx + 1 + forward <= size - 1, forward, size - 2 - idx)

        start_arr = idx - bw_arr
        end_arr = idx + 1 + fw_arr

        cols = np.arange(size)
        update_mask = (cols >= start_arr[:, None]) & (cols < end_arr[:, None])

        vals = tmp_sub[update_mask]
        tmp_sub[update_mask] = np.where(vals < pcut, vals, vals * decay_rate)

        tmp[mask] = tmp_sub

    if not norm:
        return tmp

    processed_dist = np.copy(unprocessed_dist)

    if tmp_sub.shape[0] > 0:
        norm_sub = tmp_sub / np.sum(tmp_sub, axis=-1, keepdims=True)

        if smooth:
            norm_sub = gaussian_filter1d(norm_sub, sigma, axis=-1, mode='reflect')

        processed_dist[mask] = norm_sub

    return processed_dist


_M_CACHE = {}


def convert_dgram_37_to_39(dgram2):
    """Convert a 37-bin distogram to the model's 39-bin format via a static map."""
    device, dtype = dgram2.device, dgram2.dtype
    cache_key = (device, dtype)

    if cache_key not in _M_CACHE:
        M = torch.zeros((37, 39), dtype=dtype, device=device)
        start_3, step_3 = 3.25, 1.25

        # Out-of-bounds probability (index 0) goes to the target OOB bin (index 38).
        M[0, 38] = 1.0

        for i in range(1, 37):
            mid = 2.25 + (i - 1) * 0.5

            if mid < start_3:
                continue
            elif mid >= 50.75:
                M[i, 38] = 1.0
            else:
                target_idx = int((mid - start_3) / step_3)
                M[i, target_idx] = 1.0

        _M_CACHE[cache_key] = M

    return torch.matmul(dgram2, _M_CACHE[cache_key])


def dist_to_39(dists):
    """One-hot encode pairwise distances into the model's 39 distance bins."""
    device = dists.device
    lower = torch.linspace(3.25, 50.75, 39, device=dists.device)
    dists = dists.unsqueeze(-1)
    inf = 1e8
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)
    dgram = dgram.to(device)
    return dgram


def dist_to_37(dists, device):
    """One-hot encode pairwise distances into 37 bins spanning [2.0, 20.0) + OOB."""
    dists = dists.to(device, non_blocking=True)
    if dists.ndim == 2:
        dists = dists.unsqueeze(0)
    bins_dist = torch.arange(2.0, 20.5, 0.5, device=device, dtype=dists.dtype)
    Jdist = torch.bucketize(dists, bins_dist, right=True)
    mask = (Jdist == 0) | (Jdist >= 37)
    Jdist[mask] = 0
    sdist = F.one_hot(Jdist, num_classes=37)
    return sdist

"""Cross-process serialization for parallel workers.

GPU tensors never cross process boundaries; intermediate representations,
distograms and pseudo-atom coordinates are exchanged as npz files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def reprs_to_npz(mid_data: dict, path) -> Path:
    """Save ``get_repr`` output (pred_gemo + reprs + raw_seq) to an npz file."""
    path = Path(path)
    data = {
        "raw_seq": np.array(mid_data["raw_seq"]),
        "name": np.array(mid_data["name"]),
    }
    for key in ("dist", "theta", "omega", "phi"):
        data[f"pred_gemo_{key}"] = mid_data["pred_gemo"][key].cpu().float().numpy()
    for key in ("pair", "single", "msa"):
        data[f"reprs_{key}"] = mid_data["reprs"][key].cpu().float().numpy()
    np.savez_compressed(path, **data)
    return path


def reprs_from_npz(path, device) -> dict:
    """Restore a ``get_repr``-style dict on ``device`` from a ``reprs_to_npz`` file."""
    loaded = np.load(path)
    mid_data = {
        "pred_gemo": {
            k: torch.from_numpy(loaded[f"pred_gemo_{k}"]).to(device)
            for k in ("dist", "theta", "omega", "phi")
        },
        "reprs": {
            k: torch.from_numpy(loaded[f"reprs_{k}"]).to(device)
            for k in ("pair", "single", "msa")
        },
        "raw_seq": str(loaded["raw_seq"]),
        "name": str(loaded["name"]),
    }
    return mid_data


def pseudo_beta_to_npz(pseudo_beta: torch.Tensor, path) -> Path:
    """Save pseudo-atom coordinates as a .npy file (``np.save`` appends ``.npy``,
    so callers should pass a ``.npy`` path to keep the filename exact)."""
    path = Path(path)
    np.save(path, pseudo_beta.cpu().numpy())
    return path


def pseudo_beta_from_npz(path, device) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).to(device)

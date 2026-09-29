"""Output writing: reproducible input.json, per-sample info.json, optional repr npz."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .config import REPO, EnvConfig, RunOptions, SampleConfig, asdict_safe, env_to_dict


def _portable_paths(value):
    """Represent repository-local absolute paths relative to the repository."""
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(REPO).as_posix()
            except ValueError:
                pass
    return value


def write_input_json(
    sample_dir: Path, sample: SampleConfig, options: RunOptions, env: EnvConfig
) -> None:
    """Reproducible snapshot of everything that shaped this sample's output."""
    data = {
        "sample": asdict_safe(sample),
        "options": asdict_safe(options),
        "env": env_to_dict(env),
    }
    with (sample_dir / "input.json").open("w", encoding="utf-8") as handle:
        json.dump(_portable_paths(data), handle, indent=2)
        handle.write("\n")


def write_info_json(sample_dir: Path, info: dict) -> None:
    with (sample_dir / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(_portable_paths(info), handle, indent=2)
        handle.write("\n")


def save_repr_npz(sample_dir: Path, sample_name: str, mid_data_by_model: Dict[str, dict]) -> Path:
    """Save per-model representations + distograms to ``{sample}_reprs.npz``."""
    npz_data = {}
    for model, mid in mid_data_by_model.items():
        for key in ("pair", "single", "msa"):
            npz_data[f"{model}_{key}"] = mid["reprs"][key].cpu().float().numpy()
        for key in ("dist", "theta", "omega", "phi"):
            npz_data[f"{model}_{key}"] = mid["pred_gemo"][key].cpu().float().numpy()
    out = sample_dir / f"{sample_name}_reprs.npz"
    np.savez_compressed(out, **npz_data)
    return out

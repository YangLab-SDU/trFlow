"""Runtime validation and path resolution for user-supplied configs."""
from __future__ import annotations

import os
from pathlib import Path

from .config import EnvConfig, SampleConfig, UserConfig


def resolve(path: str) -> str:
    """Resolve a path to absolute (relative paths resolve against CWD)."""
    p = Path(path)
    return str(p.resolve()) if p.is_absolute() else str((Path.cwd() / p).resolve())


def read_fasta_sequence(fasta_path: str) -> str:
    with open(fasta_path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith(">")]
    return "".join(lines)


def validate_sample(sample: SampleConfig) -> SampleConfig:
    """Resolve and check a sample's files. Returns the sample with absolute paths."""
    sample.fasta_path = resolve(sample.fasta_path)
    sample.msa_path = resolve(sample.msa_path)
    if not os.path.isfile(sample.fasta_path):
        raise FileNotFoundError(f"[{sample.name}] fasta not found: {sample.fasta_path}")
    if not os.path.isfile(sample.msa_path):
        raise FileNotFoundError(f"[{sample.name}] msa not found: {sample.msa_path}")
    if sample.init_pdb:
        sample.init_pdb = resolve(sample.init_pdb)
        if not os.path.isfile(sample.init_pdb):
            raise FileNotFoundError(f"[{sample.name}] init_pdb not found: {sample.init_pdb}")
    return sample


def validate_user_config(cfg: UserConfig) -> UserConfig:
    if not cfg.samples:
        raise ValueError("input config contains no samples")
    if not cfg.output_dir:
        raise ValueError("output_dir must not be empty")
    if cfg.options.sample_num < 1:
        raise ValueError("sample_num must be at least 1")
    if cfg.options.steps < 1:
        raise ValueError("steps must be at least 1")
    if not cfg.options.models:
        raise ValueError("at least one model must be selected")
    cfg.output_dir = resolve(cfg.output_dir)
    for sample in cfg.samples:
        validate_sample(sample)
    return cfg


def validate_env(env: EnvConfig) -> EnvConfig:
    missing = []
    for label, p in (
        ("Xray checkpoint", env.trx2flow_xray),
        ("NMR checkpoint", env.trx2flow_nmr),
        ("ESM weights", env.esm_weights),
        ("OpenFold runner", env.openfold.runner),
        ("OpenFold params", env.openfold.param_path),
    ):
        if not os.path.isfile(p):
            missing.append(f"{label}: {p}")
    if missing:
        raise FileNotFoundError(
            "environment config references missing files:\n  " + "\n  ".join(missing)
        )
    return env

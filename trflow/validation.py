"""Runtime validation and path resolution for user-supplied configs."""
from __future__ import annotations

import os
from pathlib import Path

from .config import EnvConfig, SampleConfig, UserConfig


_TARGET_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def resolve(path: str) -> str:
    """Resolve a path to absolute (relative paths resolve against CWD)."""
    p = Path(path)
    return str(p.resolve()) if p.is_absolute() else str((Path.cwd() / p).resolve())


def read_fasta_sequence(fasta_path: str) -> str:
    _header, sequence = _read_first_record(fasta_path, "FASTA")
    return _validate_target_sequence(sequence, f"FASTA file {fasta_path}")


def read_a3m_query_sequence(msa_path: str) -> tuple[str, str]:
    """Return the header and sequence for a strict single-chain A3M query.

    A3M is defined relative to its first (query) record. trFlow requires that
    record to be the ungapped target sequence; lowercase insertions and gap
    characters in the query are rejected instead of being silently discarded.
    """
    header, sequence = _read_first_record(msa_path, "A3M")
    if any(char.islower() for char in sequence):
        raise ValueError(
            f"A3M query contains lowercase insertions: {msa_path}; "
            "the first record must be the ungapped target sequence"
        )
    if "-" in sequence or "." in sequence:
        raise ValueError(
            f"A3M query contains gap characters: {msa_path}; "
            "the first record must be the ungapped target sequence"
        )
    return header, _validate_target_sequence(sequence, f"A3M query in {msa_path}")


def resolve_target_sequence(sample: SampleConfig) -> tuple[str, str, str]:
    """Resolve the target from A3M, optionally cross-checking an explicit FASTA."""
    query_header, query_sequence = read_a3m_query_sequence(sample.msa_path)
    if not sample.fasta_path:
        return query_sequence, "msa_first_record", query_header

    fasta_sequence = read_fasta_sequence(sample.fasta_path)
    if fasta_sequence != query_sequence:
        mismatch = _first_mismatch(fasta_sequence, query_sequence)
        raise ValueError(
            f"[{sample.name}] FASTA sequence does not match the first A3M "
            f"record ({mismatch}): {sample.fasta_path} vs {sample.msa_path}"
        )
    return fasta_sequence, "fasta_validated_against_msa", query_header


def _read_first_record(path: str, format_name: str) -> tuple[str, str]:
    header = None
    sequence_parts = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{format_name} has an empty first header: {path}")
            elif header is None:
                raise ValueError(
                    f"{format_name} sequence appears before its first header "
                    f"at line {line_number}: {path}"
                )
            else:
                sequence_parts.append(line)

    if header is None:
        raise ValueError(f"{format_name} contains no sequence records: {path}")
    if not sequence_parts:
        raise ValueError(f"{format_name} first record has no sequence: {path}")
    return header, "".join(sequence_parts)


def _validate_target_sequence(sequence: str, label: str) -> str:
    normalized = sequence.upper()
    invalid = sorted(set(normalized) - _TARGET_ALPHABET)
    if invalid:
        raise ValueError(f"{label} contains invalid residue characters: {''.join(invalid)}")
    if not normalized:
        raise ValueError(f"{label} is empty")
    return normalized


def _first_mismatch(left: str, right: str) -> str:
    for index, (left_residue, right_residue) in enumerate(zip(left, right), start=1):
        if left_residue != right_residue:
            return f"position {index}: FASTA={left_residue}, A3M={right_residue}"
    return f"length: FASTA={len(left)}, A3M={len(right)}"


def validate_sample(sample: SampleConfig) -> SampleConfig:
    """Resolve and check a sample's files. Returns the sample with absolute paths."""
    sample.msa_path = resolve(sample.msa_path)
    if not os.path.isfile(sample.msa_path):
        raise FileNotFoundError(f"[{sample.name}] msa not found: {sample.msa_path}")
    if sample.fasta_path:
        sample.fasta_path = resolve(sample.fasta_path)
        if not os.path.isfile(sample.fasta_path):
            raise FileNotFoundError(f"[{sample.name}] fasta not found: {sample.fasta_path}")
    if sample.init_pdb:
        sample.init_pdb = resolve(sample.init_pdb)
        if not os.path.isfile(sample.init_pdb):
            raise FileNotFoundError(f"[{sample.name}] init_pdb not found: {sample.init_pdb}")
    resolve_target_sequence(sample)
    return sample


def validate_user_config(cfg: UserConfig) -> UserConfig:
    if not cfg.samples:
        raise ValueError("input config contains no samples")
    if not cfg.output_dir:
        raise ValueError("output_dir must not be empty")
    if cfg.options.sample_num < 1:
        raise ValueError("sample_num must be at least 1")
    if cfg.options.steps < 2:
        raise ValueError("steps must be at least 2")
    if not cfg.options.models:
        raise ValueError("at least one model must be selected")
    cfg.output_dir = resolve(cfg.output_dir)
    for sample in cfg.samples:
        validate_sample(sample)
    return cfg


def validate_env(env: EnvConfig) -> EnvConfig:
    missing = []
    for label, p in (
        ("Xray checkpoint", env.trflow_xray),
        ("NMR checkpoint", env.trflow_nmr),
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

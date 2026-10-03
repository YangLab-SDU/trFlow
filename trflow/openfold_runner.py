"""OpenFold init-structure runner (Stage 1).

Runs the vendored inference script ``openfold/run_pretrained_openfold_multi.py``
(Apache-2.0, copied into this repo) as a subprocess using the OpenFold conda
env interpreter configured in ``env.openfold.python``. That interpreter (Python
3.7 / torch 1.13) is required because the vendored OpenFold does not import
under the main PyTorch env.

The script reads ``{pid}.fasta`` and ``{pid}.a3m`` from the ``--fasta_path`` /
``--msa_path`` directories, so we stage the sample's files under
``{sample_dir}/openfold_work/`` and collect
``{out}/{pid}/{model_name}_unrelaxed.pdb``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .config import EnvConfig, SampleConfig
from .console import detail


class OpenFoldError(RuntimeError):
    """Raised when the OpenFold subprocess fails."""


def run_openfold(sample: SampleConfig, env: EnvConfig, sample_dir: Path, gpu_idx: int, seed: int) -> str:
    """Run OpenFold on ``sample``; returns the path of the init structure PDB.

    The produced PDB is copied to ``{sample_dir}/{sample.name}_init.pdb`` and
    the subprocess's full log is kept at ``{sample_dir}/openfold_log.txt``.
    ``seed`` is passed as ``--data_random_seed``: OpenFold's feature pipeline
    samples/masks the MSA with the global torch/np RNG, so a fixed seed makes
    the init structure reproducible across runs.
    """
    name = sample.name
    staging = sample_dir / "openfold_work"
    out_dir = staging / "out"
    staging.mkdir(parents=True, exist_ok=True)

    # -- stage inputs ------------------------------------------------
    fasta = staging / f"{name}.fasta"
    _write_fasta(fasta, sample, name)
    shutil.copy(sample.msa_path, staging / f"{name}.a3m")
    (staging / "lst.txt").write_text(f"{name}\n")

    # -- run ---------------------------------------------------------
    # Empty openfold.python means "run with the current environment's python"
    # (single-environment setup). Otherwise the configured interpreter wins.
    interpreter = env.openfold.python or sys.executable
    cmd = [
        interpreter,
        env.openfold.runner,
        "--lst", str(staging / "lst.txt"),
        "--fasta_path", str(staging),
        "--msa_path", str(staging),
        "--output_dir", str(out_dir),
        "--model_name", env.openfold.model_name,
        "--param_path", env.openfold.param_path,
        "--gpu", str(gpu_idx),
        "--data_random_seed", str(seed),
        "--recalc",
    ]
    log_path = sample_dir / "openfold_log.txt"
    detail(f"OpenFold Python: {interpreter} | GPU: {gpu_idx} | log: {log_path}")
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=env.openfold.timeout_seconds)
    except subprocess.TimeoutExpired:
        raise OpenFoldError(
            f"[{name}] OpenFold timed out after {env.openfold.timeout_seconds}s; log: {log_path}"
        )
    if proc.returncode != 0:
        raise OpenFoldError(f"[{name}] OpenFold failed (rc={proc.returncode}); log tail:\n{_tail(log_path)}")

    # -- collect result ----------------------------------------------
    pred_pdb = out_dir / name / f"{env.openfold.model_name}_unrelaxed.pdb"
    if not pred_pdb.is_file():
        raise OpenFoldError(f"[{name}] OpenFold produced no PDB at {pred_pdb}; log tail:\n{_tail(log_path)}")
    init_pdb = sample_dir / f"{name}_init.pdb"
    shutil.copy(pred_pdb, init_pdb)
    shutil.rmtree(staging, ignore_errors=True)  # staging workdir; keep openfold_log.txt
    return str(init_pdb)


def _write_fasta(dst: Path, sample: SampleConfig, name: str) -> None:
    from .validation import resolve_target_sequence

    seq, _sequence_source, _query_header = resolve_target_sequence(sample)
    dst.write_text(f">{name}\n{seq}\n")


def _tail(path: Path, n: int = 25) -> str:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "(log unreadable)"
    return "\n".join(lines[-n:])

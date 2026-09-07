"""Evaluate predicted structures against native references with TM-score."""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .console import panel, print_logo, success


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TMSCORE = REPO / "bin" / "TMscore"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared evaluation options to a CLI parser."""
    parser.add_argument(
        "--pred-dir", "--pred_dir", dest="pred_dir", required=True,
        help="predicted PDB directory or a single PDB file",
    )
    parser.add_argument(
        "--native-dir", "--native_dir", dest="native_dir", required=True,
        help="native/reference PDB directory or a single PDB file",
    )
    parser.add_argument(
        "--output", default=None,
        help="output CSV (default: <pred-dir>/compare_summary.csv)",
    )
    parser.add_argument(
        "--tmscore", default=None,
        help="TMscore executable (default: bin/TMscore in the repository)",
    )


def find_tmscore(exe: str | None) -> str:
    path = Path(exe).expanduser() if exe else DEFAULT_TMSCORE
    path = path.resolve()
    if not path.is_file():
        source = "configured path" if exe else "repository default"
        raise FileNotFoundError(
            f"TMscore not found at {source}: {path}; "
            "pass --tmscore /path/to/TMscore to override it"
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(f"TMscore is not executable: {path}")
    return str(path)


def score_pair(tmscore: str, pred: Path, native: Path) -> Tuple[float, float]:
    """Score one pair using TMscore's sequence-aligned mode."""
    cmd = [tmscore, str(pred), str(native), "-seq"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"TMscore failed for {pred} versus {native} "
            f"(exit code {result.returncode}): {message}"
        )
    m_rmsd = re.search(r"RMSD of\s+the common residues=\s+([\d\.]+)", result.stdout)
    m_tm = re.search(r"TM-score\s+=\s+([\d\.]+)", result.stdout)
    if m_rmsd and m_tm:
        return float(m_rmsd.group(1)), float(m_tm.group(1))
    raise RuntimeError(
        f"could not parse TMscore output for {pred} versus {native}"
    )


def collect_pdbs(path: str) -> List[Path]:
    """Return one PDB file or every PDB found recursively in a directory."""
    resolved = Path(path)
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"no such path: {path}")
    pdbs = sorted(resolved.rglob("*.pdb"))
    if not pdbs:
        raise ValueError(f"no .pdb files found under: {path}")
    return pdbs


def compare(pred_dir: str, native_dir: str, tmscore: str) -> np.ndarray:
    """Return pairwise RMSD/TM-score values and per-prediction summaries."""
    preds = collect_pdbs(pred_dir)
    natives = collect_pdbs(native_dir)
    scores = np.full((len(preds), len(natives), 2), np.nan)
    for pred_index, pred in enumerate(preds):
        for native_index, native in enumerate(natives):
            rmsd, tm_score = score_pair(tmscore, pred, native)
            scores[pred_index, native_index] = (rmsd, tm_score)

    rmsds = scores[:, :, 0]
    tm_scores = scores[:, :, 1]
    summary = np.column_stack([
        np.nanmin(rmsds, axis=1),
        np.nanmax(tm_scores, axis=1),
        np.nanmean(rmsds, axis=1),
        np.nanmean(tm_scores, axis=1),
    ])
    return np.hstack([rmsds, tm_scores, summary])


def write_csv(path: Path, preds: List[Path], natives: List[Path], data: np.ndarray) -> None:
    native_names = [native.stem for native in natives]
    headers = (
        [f"{name}_RMSD" for name in native_names]
        + [f"{name}_TMscore" for name in native_names]
        + ["Min_RMSD", "Max_TMscore", "Mean_RMSD", "Mean_TMscore"]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pred"] + headers)
        for name, row in zip([pred.stem for pred in preds], data):
            writer.writerow([name] + [round(value, 4) for value in row])


def run(args: argparse.Namespace) -> int:
    """Run evaluation from parsed command-line arguments."""
    executable = find_tmscore(args.tmscore)
    preds = collect_pdbs(args.pred_dir)
    natives = collect_pdbs(args.native_dir)
    data = compare(args.pred_dir, args.native_dir, executable)

    pred_path = Path(args.pred_dir)
    default_parent = pred_path if pred_path.is_dir() else pred_path.parent
    output = Path(args.output) if args.output else default_parent / "compare_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, preds, natives, data)

    best_index = int(np.nanargmax(data[:, -3]))
    panel("Evaluation summary", [
        ("Comparisons", f"{len(preds)} predicted x {len(natives)} reference"),
        ("Best structure", preds[best_index].name),
        ("Best Min RMSD", f"{data[best_index, -4]:.3f}"),
        ("Best Max TM", f"{data[best_index, -3]:.3f}"),
        ("Mean Min RMSD", f"{np.nanmean(data[:, -4]):.3f}"),
        ("Mean Max TM", f"{np.nanmean(data[:, -3]):.3f}"),
    ])
    success(f"Evaluation written to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Compare predicted PDBs against native references with TM-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Entry point retained for the standalone evaluation script."""
    print_logo()
    return run(build_parser().parse_args(argv))

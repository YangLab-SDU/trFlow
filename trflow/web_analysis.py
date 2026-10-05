"""Align and cluster a predicted conformation ensemble for the local web UI.

All geometry uses corresponding C-alpha atoms and reports distances in Angstrom.
Pairwise distances use independent best-fit Kabsch alignments; displayed PDBs
share the coordinate frame of the first structure. Clusters use average linkage
and medoids, so their representatives are actual predicted conformations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


_DLL_DIRECTORY_HANDLES = []


def _prepare_native_runtime() -> None:
    """Make this interpreter's Conda BLAS/LAPACK DLLs available on Windows.

    Directly starting a Conda python.exe does not activate its Library/bin.
    NumPy can import successfully and fail later when LAPACK is delay-loaded.
    Some BLAS runtimes use LoadLibrary themselves, so adding a DLL directory
    alone is insufficient; the process-local PATH must include that directory.
    """
    if os.name != "nt":
        return
    library_bin = Path(sys.prefix) / "Library" / "bin"
    if not library_bin.is_dir():
        return
    normalized = os.path.normcase(os.path.abspath(library_bin))
    path_entries = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and os.path.normcase(os.path.abspath(entry)) != normalized
    ]
    os.environ["PATH"] = os.pathsep.join([str(library_bin), *path_entries])
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(library_bin)))


_prepare_native_runtime()

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import squareform


_CACHE_VERSION = 2


def _write_json(path: Path, value: Any) -> None:
    """Replace a result atomically so concurrent readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _read_pdb(path: Path) -> tuple[list[str], dict[tuple[str, str, str, str], np.ndarray], list[float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    coordinates: dict[tuple[str, str, str, str], np.ndarray] = {}
    confidence: list[float] = []
    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if line[16:17] not in (" ", "A"):
            continue
        key = (line[21:22], line[22:26].strip(), line[26:27], line[17:20].strip())
        if key in coordinates:
            continue
        try:
            position = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError as error:
            raise ValueError(f"Invalid C-alpha coordinates in {path.name}") from error
        if not np.isfinite(position).all():
            raise ValueError(f"Non-finite C-alpha coordinates in {path.name}")
        coordinates[key] = position
        try:
            value = float(line[60:66])
            if math.isfinite(value):
                confidence.append(value)
        except ValueError:
            pass
    if not coordinates:
        raise ValueError(f"No C-alpha atoms found in {path.name}")
    return lines, coordinates, confidence


def _kabsch(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return row-vector rotation/translation and a best-fit C-alpha RMSD."""
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    moving_zero = moving - moving_center
    reference_zero = reference - reference_center
    left, _, right = np.linalg.svd(moving_zero.T @ reference_zero)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(left @ right) >= 0 else -1.0
    rotation = left @ correction @ right
    translation = reference_center - moving_center @ rotation
    residual = moving @ rotation + translation - reference
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rotation, translation, rmsd


def _write_aligned_pdb(path: Path, lines: list[str], rotation: np.ndarray, translation: np.ndarray) -> None:
    """Apply the CA-derived rigid transformation to all ATOM/HETATM records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for line in lines:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    position = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError as error:
                    raise ValueError(f"Invalid atom coordinates in {path.name}") from error
                transformed = position @ rotation + translation
                line = (
                    line[:30]
                    + f"{transformed[0]:8.3f}{transformed[1]:8.3f}{transformed[2]:8.3f}"
                    + line[54:]
                )
            stream.write(line)
    os.replace(temporary, path)


def _embedding(aligned: np.ndarray) -> tuple[list[dict[str, float | int]], list[float]]:
    count, length, _ = aligned.shape
    if count == 1:
        return [{"index": 0, "x": 0.0, "y": 0.0}], [0.0, 0.0]
    flattened = aligned.reshape(count, -1)
    centered = (flattened - flattened.mean(axis=0)) / np.sqrt(length)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = eigh(gram, subset_by_index=[max(count - 2, 0), count - 1])
    eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
    eigenvectors = eigenvectors[:, ::-1]
    for axis in range(eigenvectors.shape[1]):
        extreme = np.argmax(np.abs(eigenvectors[:, axis]))
        if eigenvectors[extreme, axis] < 0:
            eigenvectors[:, axis] *= -1
    projected = eigenvectors * np.sqrt(eigenvalues)[None, :]
    if projected.shape[1] == 1:
        projected = np.column_stack((projected, np.zeros(count)))
    total = float(np.trace(gram))
    variance = [float(value / total) if total > 1e-12 else 0.0 for value in eigenvalues]
    variance += [0.0] * (2 - len(variance))
    points = [
        {"index": index, "x": round(float(point[0]), 6), "y": round(float(point[1]), 6)}
        for index, point in enumerate(projected)
    ]
    return points, variance


def _fingerprint(files: list[Path], metadata: dict | None) -> dict:
    return {
        "version": _CACHE_VERSION,
        "files": [
            {"file": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in files
        ],
        "metadata": (metadata or {}).get("predictions", []),
    }


def _prediction_files(prediction_dir: Path, metadata: dict | None) -> list[Path]:
    """Preserve prediction order, including numeric suffixes above 999."""
    metadata_order = {}
    for entry in (metadata or {}).get("predictions", []):
        try:
            metadata_order[entry["file"]] = int(entry["index"])
        except (KeyError, TypeError, ValueError):
            continue

    def order(path: Path) -> tuple:
        natural = tuple(int(token) if token.isdigit() else token.casefold()
                        for token in re.split(r"(\d+)", path.name))
        return (0, metadata_order[path.name], natural) if path.name in metadata_order else (1, 0, natural)

    return sorted(prediction_dir.glob("*.pdb"), key=order)


def analyze_ensemble(
    prediction_dir: Path, analysis_dir: Path, metadata: dict | None = None
) -> dict:
    """Write aligned PDBs and cache geometry, then return the default clustering.

    Existing caches are reused when source files and prediction metadata match.
    ``metadata`` may be the pipeline's info.json dict. The UI defaults to ten
    clusters, capped by the number of structures.
    """
    prediction_dir = Path(prediction_dir)
    analysis_dir = Path(analysis_dir)
    files = _prediction_files(prediction_dir, metadata)
    if not files:
        raise ValueError("No predicted PDB structures are available for analysis")
    fingerprint = _fingerprint(files, metadata)
    base_path = analysis_dir / "analysis.json"
    if base_path.exists():
        try:
            cached = json.loads(base_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and all(
                (analysis_dir / entry["aligned_file"]).exists() for entry in cached["structures"]
            ):
                return cluster_ensemble(analysis_dir, 10)
        except (ValueError, KeyError, OSError):
            pass

    parsed = [_read_pdb(path) for path in files]
    common = set(parsed[0][1])
    for _, atoms, _ in parsed[1:]:
        common.intersection_update(atoms)
    keys = [key for key in parsed[0][1] if key in common]
    if len(keys) < 3:
        raise ValueError("At least three corresponding C-alpha atoms are required to align the ensemble")
    coordinates = np.stack([np.stack([atoms[key] for key in keys]) for _, atoms, _ in parsed])
    count = len(files)
    aligned = np.empty_like(coordinates)
    distances = np.zeros((count, count), dtype=np.float64)
    prediction_info = {entry.get("file"): entry for entry in (metadata or {}).get("predictions", [])}
    structures = []
    for index, (path, (lines, _, confidence)) in enumerate(zip(files, parsed)):
        rotation, translation, rmsd = _kabsch(coordinates[index], coordinates[0])
        aligned[index] = coordinates[index] @ rotation + translation
        aligned_file = f"aligned/{path.name}"
        _write_aligned_pdb(analysis_dir / aligned_file, lines, rotation, translation)
        info = prediction_info.get(path.name, {})
        mean_plddt = info.get("mean_plddt")
        if mean_plddt is None and confidence:
            mean_plddt = float(np.mean(confidence)) / 100.0
        if mean_plddt is not None:
            mean_plddt = float(mean_plddt)
            if mean_plddt > 1:
                mean_plddt /= 100.0
        structures.append({
            "index": index,
            "file": path.name,
            "filename": path.name,
            "name": path.stem,
            "aligned_file": aligned_file,
            "rmsd_to_reference": round(rmsd, 6),
            "mean_plddt": mean_plddt,
            "confidence": round(mean_plddt * 100, 2) if mean_plddt is not None else None,
            "model": info.get("model"),
            "init_source": info.get("init_source"),
        })
        for earlier in range(index):
            _, _, pair_rmsd = _kabsch(coordinates[index], coordinates[earlier])
            distances[index, earlier] = distances[earlier, index] = pair_rmsd

    points, variance = _embedding(aligned)
    tree = linkage(squareform(distances, checks=False), method="average") if count > 1 else np.empty((0, 4))
    base = {
        "fingerprint": fingerprint,
        "count": count,
        "n": count,
        "length": len(keys),
        "residue_counts": [len(atoms) for _, atoms, _ in parsed],
        "reference": {"index": 0, "file": files[0].name},
        "structures": structures,
        "embedding": points,
        "embedding_axes": ["PC1", "PC2"],
        "embedding_variance_ratio": variance,
        "distance_matrix": distances.round(6).tolist(),
        "linkage": tree.tolist(),
        "method": {
            "alignment": "C-alpha Kabsch to the first predicted structure",
            "distance": "Pairwise best-fit C-alpha RMSD (Angstrom)",
            "clustering": "Average-linkage hierarchical clustering",
            "representative": "Medoid (minimum total within-cluster RMSD)",
            "embedding": "PCA of aligned C-alpha coordinates",
        },
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=analysis_dir, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, distances, allow_pickle=False)
    os.replace(temporary, analysis_dir / "distances.npy")
    _write_json(base_path, base)
    return cluster_ensemble(analysis_dir, 10)


def cluster_ensemble(analysis_dir: Path, k: int = 10) -> dict:
    """Re-cut cached hierarchy without re-aligning or recomputing pairwise RMSD."""
    analysis_dir = Path(analysis_dir)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("The number of clusters must be a positive integer")
    base = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    count = base["count"]
    effective_k = min(int(k), count)
    cluster_path = analysis_dir / f"clusters_{effective_k}.json"
    if cluster_path.exists():
        try:
            cached = json.loads(cluster_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == base["fingerprint"]:
                cached["requested_k"] = int(k)
                return cached
        except (ValueError, KeyError, OSError):
            pass
    distances = np.load(analysis_dir / "distances.npy", allow_pickle=False)
    tree = np.array(base["linkage"], dtype=np.float64)
    raw_labels = cut_tree(tree, n_clusters=[effective_k]).ravel() if count > 1 else np.zeros(1, dtype=int)
    groups = sorted(
        (np.flatnonzero(raw_labels == label).tolist() for label in np.unique(raw_labels)),
        key=lambda members: members[0],
    )
    assignments = np.zeros(count, dtype=int)
    representatives = set()
    clusters = []
    for cluster_id, members in enumerate(groups, start=1):
        submatrix = distances[np.ix_(members, members)]
        representative = members[int(np.argmin(submatrix.sum(axis=1)))]
        representatives.add(representative)
        assignments[members] = cluster_id
        pairs = submatrix[np.triu_indices(len(members), 1)]
        clusters.append({
            "id": cluster_id,
            "count": len(members),
            "size": len(members),
            "members": members,
            "representative": representative,
            "representative_file": base["structures"][representative]["file"],
            "mean_rmsd": round(float(pairs.mean()), 6) if len(pairs) else 0.0,
            "max_rmsd": round(float(pairs.max()), 6) if len(pairs) else 0.0,
        })
    for index, structure in enumerate(base["structures"]):
        structure["cluster"] = int(assignments[index])
        structure["representative"] = index in representatives
    for point in base["embedding"]:
        point["cluster"] = int(assignments[point["index"]])
        point["filename"] = base["structures"][point["index"]]["file"]
    result = {
        **base,
        "k": effective_k,
        "requested_k": int(k),
        "clusters": clusters,
    }
    _write_json(cluster_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run native analysis in an isolated worker, separate from GPU inference."""
    parser = argparse.ArgumentParser(description="Align and cluster a trFlow conformation ensemble")
    parser.add_argument("--predictions", type=Path, help="Prediction PDB directory; omit to re-cut an existing cache")
    parser.add_argument("--analysis", type=Path, required=True, help="Analysis cache and aligned PDB directory")
    parser.add_argument("--metadata", type=Path, help="Pipeline info.json metadata")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters (default: 10)")
    parser.add_argument("--result", type=Path, help="Write result JSON atomically to this file")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be a positive integer")
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None
        if args.predictions:
            analyze_ensemble(args.predictions, args.analysis, metadata)
        result = cluster_ensemble(args.analysis, args.k)
        if args.result:
            _write_json(args.result, result)
        else:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Ensemble analysis failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

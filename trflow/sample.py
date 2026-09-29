"""Stage 3: sample generation along each model/init-source path.

Paths (per the original ``main_v3`` eval setup, all using the V6 structure
model with ``flow_mode='iterative'``):

  * geometric exploration ON  -> 4 paths: Xray·init, Xray·updated, NMR·init, NMR·updated
  * geometric exploration OFF -> 2 paths: Xray·init, NMR·init
  * single model              -> half as many paths

``sample_num`` is distributed round-robin: ``sample_num // n_paths`` each, with
the remainder going to the first paths.
"""
from __future__ import annotations

from typing import List, Tuple

from .core import FlowInferenceCore, Inf_StruData

INIT_SOURCE = "init"
UPDATED_SOURCE = "updated"


def build_paths(models: List[str], geometric_exploration: bool) -> List[Tuple[str, str]]:
    """Return ``(model, init_source)`` path tuples."""
    paths = []
    for model in models:
        paths.append((model, INIT_SOURCE))
        if geometric_exploration:
            paths.append((model, UPDATED_SOURCE))
    return paths


def allocate_k(sample_num: int, n_paths: int) -> List[int]:
    """Split ``sample_num`` into ``n_paths`` counts; remainder to the first paths."""
    base, rem = divmod(sample_num, n_paths)
    return [base + (1 if i < rem else 0) for i in range(n_paths)]


def run_path_generation(
    core: FlowInferenceCore,
    inf_strudata: Inf_StruData,
    pseudo_beta,
    prior,
    out_dir,
    name_prefix: str,
    k: int,
    steps: int,
    random_step: bool,
    random_step_size: bool,
    start_index: int,
) -> List[dict]:
    """Generate ``k`` samples along one path; returns per-sample result dicts."""
    results = []
    for i in range(k):
        index = start_index + i + 1
        name = f"{name_prefix}_sample_{index:03d}"
        pdb_path, plddt = core.sample_one(
            inf_strudata, pseudo_beta, prior, out_dir, name, steps=steps,
            random_step=random_step, random_step_size=random_step_size,
        )
        results.append({"index": index, "file": f"{name}.pdb", "path": pdb_path, "mean_plddt": plddt})
    return results

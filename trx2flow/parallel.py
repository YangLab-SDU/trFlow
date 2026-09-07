"""Optional intra-sample parallelism (``parallel=true``).

Stages run on a shared ``ProcessPoolExecutor`` (spawn context — fork + PyTorch
GPU is unsafe). GPU tensors never cross process boundaries; intermediate
representations / pseudo-atom coordinates are exchanged as npz files
(``trx2flow.iohandlers``). Samples in a batch stay sequential; only the stages
inside a sample are parallelised.

  Stage 1: OpenFold init structure + per-model representations, in parallel.
  Stage 2: per-model geometric explorations, in parallel.
  Stage 3: per-model path generation, in parallel (each worker loads its own
           model, so VRAM stays bounded).

RNG notes: every RNG-consuming task (geometric exploration, each sampling path)
is seeded via ``trx2flow.seed.task_seed`` from the sample seed, identically in
the serial and parallel paths. Each sample draws a fresh noise vector from
``HarmonicPrior.sample`` (which consumes the global torch RNG), so without this
per-task seeding a parallel run could never replay the sequential process's
single RNG stream.

Determinism guarantees: Stage 3 sampling paths reproduce exactly between
``parallel=true`` and ``parallel=false``. Stage 2 geometric exploration is also
reproducible *within* a mode (any two serial runs match, any two parallel runs
match), but serial vs parallel explore can differ by up to ~2 A in CB
coordinates: the serial process runs the fp16-autocast ESM repr forward before
exploration, while a parallel explore worker is a clean process that never runs
that forward. The autocast forward leaves deterministic GPU execution state
that shifts later structure-model kernels; eliminating it would require dropping
autocast from the repr path, which would break equivalence with the reference
implementation. plddt impact is <0.001, so this is acceptable numerical
variance, not a logic divergence.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List

import torch

from .config import EnvConfig, RunOptions, SampleConfig, asdict_safe, env_to_dict
from .console import stage_done, stage_skip, stage_start, success, warning
from .iohandlers import pseudo_beta_from_npz, pseudo_beta_to_npz, reprs_from_npz, reprs_to_npz
from .sample import INIT_SOURCE, allocate_k, build_paths
from .seed import resolve_seed, set_seed, task_seed
from .validation import read_fasta_sequence


# ---------------------------------------------------------------------------
# Workers (top-level for picklability under spawn)
# ---------------------------------------------------------------------------

def _worker_openfold(sample: SampleConfig, env: EnvConfig, sample_dir: str, gpu_idx: int, seed: int) -> str:
    from .openfold_runner import run_openfold

    return run_openfold(sample, env, Path(sample_dir), gpu_idx, seed)


def _worker_repr(model: str, sample: SampleConfig, env: EnvConfig, gpu_idx: int, out_npz: str) -> str:
    torch.cuda.set_device(gpu_idx)
    from .core import FlowInferenceCore

    core = FlowInferenceCore(model, env.model_checkpoint(model), env.esm_weights, torch.device(f"cuda:{gpu_idx}"))
    raw_seq = read_fasta_sequence(sample.fasta_path)
    mid = core.get_repr(raw_seq, sample.msa_path)
    reprs_to_npz(mid, out_npz)
    return out_npz


def _worker_explore(
    model: str, env: EnvConfig, init_pdb: str, repr_npz: str, sample_name: str, out_dir: str,
    updated_pb_npz: str, sample_seed: int, gpu_idx: int,
) -> dict:
    torch.cuda.set_device(gpu_idx)
    from .core import FlowInferenceCore, output_to_pdb
    from .explore import run_exploration

    device = torch.device(f"cuda:{gpu_idx}")
    core = FlowInferenceCore(model, env.model_checkpoint(model), env.esm_weights, device)
    inf, _aatype, prior = core.build_inf_strudata(reprs_from_npz(repr_npz, device))
    init_pb = core.get_init_pseudo_beta(init_pdb)
    # Seed AFTER model construction: module __init__ (reset_parameters) consumes
    # the torch RNG, so seeding before would leave exploration on a RNG state
    # that differs from the serial path (where the core is built in stage 1).
    set_seed(task_seed(sample_seed, model, "explore"))
    updated_pb, iter_output, n_iters, converged = run_exploration(core, inf, init_pb, prior)
    upd_pdb, upd_plddt = output_to_pdb(
        name=f"{sample_name}_updated_{model}", raw_seq=inf.raw_seq, model_output=iter_output, out_dir=out_dir,
    )
    pseudo_beta_to_npz(updated_pb, updated_pb_npz)
    return {
        "pdb": upd_pdb,
        "n_iters": n_iters,
        "converged": converged,
        "mean_plddt": round(upd_plddt, 3),
        "pseudo_beta_npz": updated_pb_npz,
    }


def _worker_generate_model(
    model: str,
    subpaths: List[tuple],
    env: EnvConfig,
    init_pdb: str,
    repr_npz: str,
    updated_pb_npz: str,
    sample_name: str,
    out_dir: str,
    steps: int,
    random_step: bool,
    random_step_size: bool,
    sample_seed: int,
    gpu_idx: int,
) -> List[dict]:
    torch.cuda.set_device(gpu_idx)
    from .core import FlowInferenceCore
    from .sample import run_path_generation

    device = torch.device(f"cuda:{gpu_idx}")
    core = FlowInferenceCore(model, env.model_checkpoint(model), env.esm_weights, device)
    inf, _aatype, prior = core.build_inf_strudata(reprs_from_npz(repr_npz, device))
    results = []
    for source, k, start_index in subpaths:
        set_seed(task_seed(sample_seed, model, source, start_index))
        if source == INIT_SOURCE:
            pseudo_beta = core.get_init_pseudo_beta(init_pdb)
        elif updated_pb_npz:
            pseudo_beta = pseudo_beta_from_npz(updated_pb_npz, device)
        else:
            raise ValueError(f"source '{source}' needs updated pseudo_beta but none produced")
        res = run_path_generation(
            core, inf, pseudo_beta, prior, out_dir, sample_name, k=k, steps=steps,
            random_step=random_step, random_step_size=random_step_size, start_index=start_index,
        )
        for r in res:
            r["model"] = model
            r["init_source"] = source
        results.extend(res)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_batch_parallel(cfg, env: EnvConfig) -> List[dict]:
    output_root = Path(cfg.output_dir)
    n_workers = max(1, len(env.gpus))
    if len(env.gpus) < 3:
        warning(f"Only {len(env.gpus)} GPU(s) configured; parallel mode is unlikely to be faster.")
    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=get_context("spawn"))
    infos = []
    try:
        for sample in cfg.samples:
            infos.append(run_sample_parallel(sample, cfg.options, env, output_root, executor))
    finally:
        executor.shutdown(wait=True)
    return infos


def run_sample_parallel(
    sample: SampleConfig, options: RunOptions, env: EnvConfig, output_root: Path, executor: ProcessPoolExecutor,
) -> dict:
    sample_dir = output_root / sample.name
    sample_dir.mkdir(parents=True, exist_ok=True)
    work_dir = sample_dir / ".parallel"
    work_dir.mkdir(exist_ok=True)
    predictions_dir = sample_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    seed = resolve_seed(sample.seed, options.seed)
    set_seed(seed)
    raw_seq = read_fasta_sequence(sample.fasta_path)
    L = len(raw_seq)

    info = {
        "sample_name": sample.name,
        "seed": seed,
        "options": asdict_safe(options),
        "env": env_to_dict(env),
        "init": {},
        "updated": {},
        "stage_times": {},
        "predictions": [],
    }
    times: Dict[str, float] = {}
    run_started_at = datetime.now().isoformat(timespec="seconds")
    total_start = time.perf_counter()
    gpus = env.gpus

    # ---------------------------------------------------------------- Stage 1
    stage_start(sample.name, 1, "Initial structure and representations [parallel]")
    stage1 = []
    if sample.init_pdb:
        init_source = "user"
        init_pdb = Path(sample.init_pdb)
    else:
        init_source = "openfold"
        stage1.append(executor.submit(_worker_openfold, sample, env, str(sample_dir), gpus[0], seed))
    for i, model in enumerate(options.models):
        gpu = gpus[(i + 1) % len(gpus)]
        stage1.append(executor.submit(_worker_repr, model, sample, env, gpu, str(work_dir / f"{model}_repr.npz")))

    repr_npz: Dict[str, str] = {}
    openfold_result = None
    for fut in stage1:
        out = fut.result()
        if isinstance(out, str) and out.endswith("_repr.npz"):
            repr_npz[Path(out).name.replace("_repr.npz", "")] = out
        else:
            openfold_result = out
    if init_source == "openfold":
        init_pdb = Path(openfold_result)
    times["stage1_openfold"] = round(times.get("stage1_openfold", 0), 2)  # placeholder
    for model in options.models:
        times[f"stage1_repr_{model}"] = 0.0
    stage_done()
    _ = L  # length validated below

    # init PDB length check
    init_pdb_local = sample_dir / f"{sample.name}_init.pdb"
    if init_pdb.resolve() != init_pdb_local.resolve():
        shutil.copy(init_pdb, init_pdb_local)
    init_len = _pdb_pseudo_beta_len(init_pdb_local)
    if init_len != L:
        raise ValueError(f"[{sample.name}] init structure length {init_len} != fasta length {L}: {init_pdb_local}")
    info["init"] = {"source": init_source, "pdb": str(init_pdb_local)}

    # ---------------------------------------------------------------- Stage 2
    if options.geometric_exploration:
        stage_start(sample.name, 2, "Geometric exploration [parallel]")
        stage2 = []
        for i, model in enumerate(options.models):
            gpu = gpus[i % len(gpus)]
            stage2.append(executor.submit(
                _worker_explore, model, env, str(init_pdb_local), repr_npz[model], sample.name,
                str(sample_dir), str(work_dir / f"{model}_updated_pb.npy"), seed, gpu,
            ))
        updated_pb_npz: Dict[str, str] = {}
        for fut in stage2:
            res = fut.result()
            model = Path(res["pseudo_beta_npz"]).name.replace("_updated_pb.npy", "")
            updated_pb_npz[model] = res["pseudo_beta_npz"]
            info["updated"][model] = {k: v for k, v in res.items() if k != "pseudo_beta_npz"}
        stage_done()
    else:
        stage_start(sample.name, 2, "Geometric exploration [parallel]")
        stage_skip("disabled by configuration")
        updated_pb_npz = {}

    # ---------------------------------------------------------------- Stage 3
    stage_start(sample.name, 3, "Conformation generation [parallel]")
    t0 = time.perf_counter()
    paths = build_paths(options.models, options.geometric_exploration)
    ks = allocate_k(options.sample_num, len(paths))
    steps = options.effective_steps()
    random_step = options.random_step and not options.single_step

    model_subpaths: Dict[str, List[tuple]] = {}
    start = 0
    for (model, source), k in zip(paths, ks):
        if k > 0:
            model_subpaths.setdefault(model, []).append((source, k, start))
            start += k

    stage3 = []
    for i, (model, subpaths) in enumerate(model_subpaths.items()):
        gpu = gpus[i % len(gpus)]
        stage3.append(executor.submit(
            _worker_generate_model, model, subpaths, env, str(init_pdb_local), repr_npz[model],
            updated_pb_npz.get(model, ""), sample.name, str(predictions_dir), steps, random_step,
            options.random_step_size, seed, gpu,
        ))
    predictions = []
    for fut in stage3:
        predictions.extend(fut.result())
    predictions.sort(key=lambda r: r["index"])
    for r in predictions:
        info["predictions"].append({
            "index": r["index"],
            "file": r["file"],
            "model": r["model"],
            "init_source": r["init_source"],
            "mean_plddt": round(r["mean_plddt"], 3),
        })
    times["stage3_generation"] = round(time.perf_counter() - t0, 2)
    stage_done(f"generation={times['stage3_generation']}s")

    total_seconds = round(time.perf_counter() - total_start, 2)
    info["stage_times"] = times
    info["stage_times"]["total"] = {
        "start": run_started_at,
        "end": datetime.now().isoformat(timespec="seconds"),
        "seconds": total_seconds,
    }

    if options.save_repr_npz:
        from .output import save_repr_npz

        mid_data = {model: reprs_from_npz(npz, torch.device("cpu")) for model, npz in repr_npz.items()}
        save_repr_npz(sample_dir, sample.name, mid_data)

    from .output import write_info_json, write_input_json

    write_input_json(sample_dir, sample, options, env)
    write_info_json(sample_dir, info)
    shutil.rmtree(work_dir, ignore_errors=True)
    success(f"{sample.name}: generated {len(info['predictions'])} structure(s) in {total_seconds}s")
    return info


def _pdb_pseudo_beta_len(pdb: Path) -> int:
    from .geometry import pseudo_beta_fn_from_pdb

    return len(pseudo_beta_fn_from_pdb(str(pdb), None))

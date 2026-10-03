"""Sequential 3-stage pipeline for one sample.

Stage 1: init structure (OpenFold subprocess or user ``init_pdb``) + per-model
         representations (Xray / NMR repr networks).
Stage 2: geometric exploration (optional) -> one updated structure per model.
Stage 3: sample generation along the model/init-source paths until ``sample_num``.

This is the default (``parallel=False``) execution path. Output layout per
sample lives under ``{output_dir}/{sample_name}/``; see ``trflow.output``.
"""
from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from .config import EnvConfig, RunOptions, SampleConfig, asdict_safe, env_to_dict
from .console import detail, stage_done, stage_skip, stage_start, success
from .core import FlowInferenceCore, output_to_pdb, parse_a3m
from .explore import run_exploration
from .openfold_runner import run_openfold
from .output import save_repr_npz, write_info_json, write_input_json
from .sample import INIT_SOURCE, allocate_k, build_paths, run_path_generation
from .seed import resolve_seed, set_seed, task_seed
from .validation import resolve_target_sequence

_MAX_EXPLORE_ITERS = 1000


def _pick_device(env: EnvConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{env.gpus[0]}")
    return torch.device("cpu")


def run_sample(sample: SampleConfig, options: RunOptions, env: EnvConfig, output_root: Path) -> dict:
    """Run the full pipeline for one sample; returns its ``info.json`` content."""
    sample_dir = output_root / sample.name
    sample_dir.mkdir(parents=True, exist_ok=True)

    seed = resolve_seed(sample.seed, options.seed)
    set_seed(seed)

    raw_seq, sequence_source, query_header = resolve_target_sequence(sample)
    L = len(raw_seq)
    _validate_msa(sample, raw_seq)

    info = {
        "sample_name": sample.name,
        "seed": seed,
        "sequence": {
            "source": sequence_source,
            "msa_query_header": query_header,
            "length": L,
        },
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
    device = _pick_device(env)

    # ------------------------------------------------------------------ Stage 1
    stage_start(sample.name, 1, "Initial structure and representations")
    if sample.init_pdb:
        init_pdb = Path(sample.init_pdb)
        init_source = "user"
    else:
        init_source = "openfold"
        detail(f"Running OpenFold ({env.openfold.model_name}) on {device}")
        t0 = time.perf_counter()
        init_pdb = Path(run_openfold(sample, env, sample_dir, env.gpus[0], seed))
        times["stage1_openfold"] = round(time.perf_counter() - t0, 2)

    init_pdb_local = sample_dir / f"{sample.name}_init.pdb"
    if init_pdb.resolve() != init_pdb_local.resolve():
        shutil.copy(init_pdb, init_pdb_local)
    init_len = _pdb_pseudo_beta_len(init_pdb_local)
    if init_len != L:
        raise ValueError(
            f"[{sample.name}] init structure length {init_len} != target length {L}: {init_pdb_local}"
        )
    info["init"] = {"source": init_source, "pdb": str(init_pdb_local)}

    cores: Dict[str, FlowInferenceCore] = {}
    mid_data: Dict[str, dict] = {}
    for model in options.models:
        t0 = time.perf_counter()
        core = FlowInferenceCore(model, env.model_checkpoint(model), env.esm_weights, device)
        cores[model] = core
        mid_data[model] = core.get_repr(raw_seq, sample.msa_path)
        times[f"stage1_repr_{model}"] = round(time.perf_counter() - t0, 2)
    stage_done(_fmt(times, "stage1_"))

    def build_model_state():
        state: Dict[str, dict] = {}
        for model in options.models:
            core = cores[model]
            inf, _aatype, prior = core.build_inf_strudata(mid_data[model])
            init_pb = core.get_init_pseudo_beta(str(init_pdb_local))
            state[model] = {"core": core, "inf": inf, "prior": prior, "init_pb": init_pb, "updated_pb": None}
        return state

    # ------------------------------------------------------------------ Stage 2
    if options.geometric_exploration:
        stage_start(sample.name, 2, "Geometric exploration")
        model_state = build_model_state()
        for model in options.models:
            t0 = time.perf_counter()
            st = model_state[model]
            # Same per-task seed as the parallel path, so explore output (which
            # draws a noise vector from prior.sample()) is identical either way.
            set_seed(task_seed(seed, model, "explore"))
            updated_pb, iter_output, n_iters, converged = run_exploration(
                st["core"], st["inf"], st["init_pb"], st["prior"], max_iters=_MAX_EXPLORE_ITERS,
            )
            upd_pdb, upd_plddt = output_to_pdb(
                name=f"{sample.name}_updated_{model}",
                raw_seq=raw_seq,
                model_output=iter_output,
                out_dir=str(sample_dir),
            )
            st["updated_pb"] = updated_pb
            info["updated"][model] = {
                "pdb": upd_pdb,
                "n_iters": n_iters,
                "converged": converged,
                "mean_plddt": round(upd_plddt, 3),
            }
            times[f"stage2_explore_{model}"] = round(time.perf_counter() - t0, 2)
        stage_done(_fmt(times, "stage2_"))
    else:
        stage_start(sample.name, 2, "Geometric exploration")
        stage_skip("disabled by configuration")
        model_state = build_model_state()

    # ------------------------------------------------------------------ Stage 3
    stage_start(sample.name, 3, "Conformation generation")
    t0 = time.perf_counter()
    predictions_dir = sample_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    paths = build_paths(options.models, options.geometric_exploration)
    ks = allocate_k(options.sample_num, len(paths))
    steps = options.effective_steps()
    random_step = options.random_step and not options.single_step
    random_step_size = options.random_step_size

    next_index = 0
    for (model, source), k in zip(paths, ks):
        if k == 0:
            continue
        st = model_state[model]
        pseudo_beta = st["init_pb"] if source == INIT_SOURCE else st["updated_pb"]
        # Per-path seed, matching the parallel workers, so every sample's noise
        # vector (prior.sample()) is independent of pipeline-internal RNG state.
        set_seed(task_seed(seed, model, source, next_index))
        results = run_path_generation(
            st["core"], st["inf"], pseudo_beta, st["prior"],
            out_dir=str(predictions_dir),
            name_prefix=sample.name,
            k=k,
            steps=steps,
            random_step=random_step,
            random_step_size=random_step_size,
            start_index=next_index,
        )
        next_index += k
        for r in results:
            info["predictions"].append({
                "index": r["index"],
                "file": r["file"],
                "model": model,
                "init_source": source,
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
        save_repr_npz(sample_dir, sample.name, mid_data)
        detail(f"Saved {sample.name}_reprs.npz")

    write_input_json(sample_dir, sample, options, env)
    write_info_json(sample_dir, info)
    success(f"{sample.name}: generated {len(info['predictions'])} structure(s) in {total_seconds}s")
    return info


def _validate_msa(sample: SampleConfig, raw_seq: str) -> None:
    msa = parse_a3m(sample.msa_path)
    if msa.shape[0] == 0:
        raise ValueError(f"[{sample.name}] empty msa: {sample.msa_path}")
    if msa.shape[1] != len(raw_seq):
        raise ValueError(
            f"[{sample.name}] msa length {msa.shape[1]} != target length {len(raw_seq)}: {sample.msa_path}"
        )


def _pdb_pseudo_beta_len(pdb: Path) -> int:
    from .geometry import pseudo_beta_fn_from_pdb

    return len(pseudo_beta_fn_from_pdb(str(pdb), None))


def _fmt(times: Dict[str, float], prefix: str) -> str:
    parts = [f"{k.replace(prefix, '')}={v}s" for k, v in times.items() if k.startswith(prefix)]
    return ", ".join(parts)

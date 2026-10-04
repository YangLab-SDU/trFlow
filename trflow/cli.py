"""Command-line interface for trFlow.

Primary interface::

    trFlow predict INPUT [options]
    trFlow evaluate --pred-dir PREDICTIONS --native-dir REFERENCES [options]

``INPUT`` may be a JSON run configuration, an A3M file, or a FASTA file.
For A3M input, the first record is the target sequence and an optional FASTA
can be supplied for strict cross-validation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .config import (
    MODEL_NAMES,
    RunOptions,
    SampleConfig,
    UserConfig,
    load_env_config,
    load_user_config,
)
from .console import panel, print_logo, success
from .validation import validate_env, validate_user_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trFlow",
        description="Sample protein conformations with trFlow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict",
        help="predict conformations from JSON, A3M, or FASTA + A3M input",
        description=(
            "Predict conformations. INPUT may be a JSON run configuration or "
            "an A3M file. FASTA input is also supported and requires --msa."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_predict_arguments(predict)
    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate predicted structures against native references",
        description="Compare predicted PDBs against native references with TM-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from .evaluation import add_arguments as add_evaluation_arguments

    add_evaluation_arguments(evaluate)
    return parser


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path", nargs="?", help="input JSON, A3M, or FASTA file")
    parser.add_argument("--input", dest="input_json", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fasta",
        help="optional FASTA to validate an A3M query (or FASTA input when used with --msa)",
    )
    parser.add_argument("--msa", help="A3M file (required for FASTA input)")
    parser.add_argument("--name", help="sample name for direct input (default: input stem)")
    parser.add_argument(
        "--init-pdb", "--init_pdb", dest="init_pdb",
        help="optional initial PDB; skips OpenFold",
    )
    parser.add_argument("--env", default=None, help="environment configuration JSON")
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", default=None,
        help="output root (direct-input default: outputs)",
    )
    parser.add_argument(
        "--sample-num", "--sample_num", dest="sample_num", type=int,
        default=argparse.SUPPRESS,
        help="number of conformations (default: 200)",
    )
    parser.add_argument("--models", default=None, help="comma-separated models: Xray,NMR")
    parser.add_argument("--parallel", action="store_true", help="parallelize intra-sample stages")
    parser.add_argument(
        "--single-step", "--single_step", dest="single_step", action="store_true",
        help="force one structure-model forward",
    )
    parser.add_argument(
        "--no-geometric-exploration", "--no-geometric_exploration",
        dest="no_geometric_exploration", action="store_true",
        help="skip geometric exploration",
    )
    parser.add_argument("--steps", type=int, default=None, help="flow schedule steps")
    parser.add_argument(
        "--no-random-step", action="store_false", dest="random_step", default=None,
        help="use --steps instead of randomly choosing 2 or 8",
    )
    parser.add_argument(
        "--no-random-step-size", action="store_false", dest="random_step_size", default=None,
        help="use evenly spaced flow steps",
    )
    parser.add_argument("--save-repr-npz", action="store_true", help="save intermediate representations")
    parser.add_argument("--seed", type=int, default=None, help="optional reproducibility seed")


def apply_cli(cfg: UserConfig, args: argparse.Namespace) -> UserConfig:
    if args.output_dir:
        cfg.output_dir = args.output_dir
    sample_num = getattr(args, "sample_num", None)
    if sample_num is not None:
        cfg.options.sample_num = sample_num
    if args.models:
        requested = [item.strip() for item in args.models.split(",") if item.strip()]
        invalid = [item for item in requested if item not in MODEL_NAMES]
        if invalid or not requested:
            raise ValueError(
                f"--models must contain only {', '.join(MODEL_NAMES)}; got: {args.models}"
            )
        cfg.options.models = requested
    if args.parallel:
        cfg.options.parallel = True
    if args.single_step:
        cfg.options.single_step = True
    if args.no_geometric_exploration:
        cfg.options.geometric_exploration = False
    if args.seed is not None:
        cfg.options.seed = args.seed
    if args.steps is not None:
        cfg.options.steps = args.steps
    if args.random_step is not None:
        cfg.options.random_step = args.random_step
    if args.random_step_size is not None:
        cfg.options.random_step_size = args.random_step_size
    if args.save_repr_npz:
        cfg.options.save_repr_npz = True
    return cfg


def _normalize_argv(argv: Optional[Sequence[str]]) -> list[str]:
    """Insert ``predict`` for the pre-subcommand CLI forms."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] not in {"predict", "evaluate", "-h", "--help"}:
        values.insert(0, "predict")
    return values


def _load_predict_config(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> tuple[UserConfig, str, str]:
    if args.input_json:
        if args.input_path:
            parser.error("use either INPUT or --input, not both")
        if args.fasta or args.msa or args.name or args.init_pdb:
            parser.error("--fasta, --msa, --name, and --init-pdb are invalid with JSON input")
        return load_user_config(args.input_json), "JSON", args.input_json

    if not args.input_path:
        if not (args.fasta and args.msa):
            parser.error("predict requires an INPUT JSON/A3M/FASTA file")
        fasta = args.fasta
        msa = args.msa
        mode = "FASTA + A3M"
        input_stem = Path(fasta).stem
        input_label = f"{fasta} + {msa}"
    else:
        suffix = Path(args.input_path).suffix.lower()
        if suffix == ".json":
            if args.fasta or args.msa or args.name or args.init_pdb:
                parser.error("--fasta, --msa, --name, and --init-pdb are invalid with JSON input")
            return load_user_config(args.input_path), "JSON", args.input_path
        if suffix in {".a3m", ".a2m"}:
            if args.msa:
                parser.error("--msa is not used when INPUT is already an A3M file")
            msa = args.input_path
            fasta = args.fasta
            mode = "A3M + FASTA validation" if fasta else "A3M"
            input_stem = Path(msa).stem
            input_label = f"{msa} + {fasta}" if fasta else msa
        else:
            if args.fasta:
                parser.error("--fasta is only used with A3M INPUT or legacy --msa mode")
            if not args.msa:
                parser.error("--msa is required when INPUT is a FASTA file")
            fasta = args.input_path
            msa = args.msa
            mode = "FASTA + A3M"
            input_stem = Path(fasta).stem
            input_label = f"{fasta} + {msa}"

    cfg = UserConfig(
        output_dir=args.output_dir or "outputs",
        samples=[SampleConfig(
            name=args.name or input_stem,
            msa_path=msa,
            fasta_path=fasta,
            seed=args.seed,
            init_pdb=args.init_pdb,
        )],
        options=RunOptions(),
    )
    return cfg, mode, input_label


def main(argv: Optional[Sequence[str]] = None) -> int:
    print_logo()
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if args.command == "evaluate":
        from .evaluation import run as run_evaluation

        return run_evaluation(args)

    cfg, mode, input_label = _load_predict_config(args, parser)
    cfg = validate_user_config(apply_cli(cfg, args))
    env = validate_env(load_env_config(args.env))

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    panel("Prediction", [
        ("Input mode", mode),
        ("Input", input_label),
        ("Output", output_root),
        ("Samples", ", ".join(sample.name for sample in cfg.samples)),
        ("Models", ", ".join(cfg.options.models)),
        ("Conformations", cfg.options.sample_num),
        ("Exploration", "on" if cfg.options.geometric_exploration else "off"),
        ("Flow", "single step" if cfg.options.single_step else f"{cfg.options.steps} steps"),
        ("Parallel", "on" if cfg.options.parallel else "off"),
        ("Seed", cfg.options.seed if cfg.options.seed is not None else "random"),
    ])

    started = time.perf_counter()
    if cfg.options.parallel:
        from .parallel import run_batch_parallel

        infos = run_batch_parallel(cfg, env)
    else:
        from .pipeline import run_sample

        infos = [run_sample(sample, cfg.options, env, output_root) for sample in cfg.samples]
    total = time.perf_counter() - started

    panel("Summary", [
        (
            info["sample_name"],
            f"{len(info['predictions'])} structure(s) in {info['stage_times']['total']['seconds']}s",
        )
        for info in infos
    ])
    success(f"Prediction finished in {total:.1f}s. Results: {output_root}")
    return 0

"""Configuration dataclasses and loaders.

Path conventions:
  * ``REPO`` is the repository root (``github/``). All relative paths in the
    environment config are resolved against ``REPO``.
  * ``env_config.json`` holds machine-specific paths (model checkpoints, ESM
    weights, OpenFold interpreter/params, GPUs). It is committed as a
    template; the ``models/`` weights themselves are git-ignored.
  * The user input JSON holds per-sample data + run options.

Merge precedence (lowest -> highest):
    dataclass defaults < env_config.json < input JSON < CLI flags
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

MODEL_NAMES = ("Xray", "NMR")


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------
@dataclass
class OpenFoldConfig:
    python: str = ""                           # empty = run with the current python
    runner: str = "openfold/run_pretrained_openfold_multi.py"
    model_name: str = "model_5_ptm"            # AlphaFold2 5th model (pTM head)
    param_path: str = "models/openfold_params_model_5_ptm.npz"
    timeout_seconds: int = 7200
    cpus: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenFoldConfig":
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EnvConfig:
    trflow_xray: str = "models/trflow_xray.pth"
    trflow_nmr: str = "models/trflow_nmr.pth"
    esm_weights: str = "models/esm_msa1_t12_100M_UR50S.pt"
    openfold: OpenFoldConfig = field(default_factory=OpenFoldConfig)
    gpus: List[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvConfig":
        d = d or {}
        return cls(
            trflow_xray=d.get("trflow_xray", cls.trflow_xray),
            trflow_nmr=d.get("trflow_nmr", cls.trflow_nmr),
            esm_weights=d.get("esm_weights", cls.esm_weights),
            openfold=OpenFoldConfig.from_dict(d.get("openfold")),
            gpus=d.get("gpus", [0]),
        )

    def resolve(self, repo: Path = REPO) -> "EnvConfig":
        """Resolve relative paths against ``repo`` (in place)."""
        for attr in ("trflow_xray", "trflow_nmr", "esm_weights"):
            p = Path(getattr(self, attr))
            if not p.is_absolute():
                setattr(self, attr, str(repo / p))
        self.openfold.runner = self._resolve(repo, self.openfold.runner)
        self.openfold.param_path = self._resolve(repo, self.openfold.param_path)
        return self

    @staticmethod
    def _resolve(repo: Path, p: str) -> str:
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str(repo / pp)

    def model_checkpoint(self, model: str) -> str:
        if model == "Xray":
            return self.trflow_xray
        if model == "NMR":
            return self.trflow_nmr
        raise ValueError(f"unknown model '{model}', expected one of {MODEL_NAMES}")


# ---------------------------------------------------------------------------
# User input config
# ---------------------------------------------------------------------------
@dataclass
class RunOptions:
    sample_num: int = 200
    geometric_exploration: bool = True
    single_step: bool = False
    steps: int = 2
    random_step: bool = True          # per sample choose step in {2, 8}
    random_step_size: bool = True     # Dirichlet smooth_steps step sizes
    models: List[str] = field(default_factory=lambda: ["Xray", "NMR"])
    parallel: bool = False
    save_repr_npz: bool = False
    seed: Optional[int] = None
    gpus: Optional[List[int]] = None
    max_workers: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RunOptions":
        d = d or {}
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        models = kwargs.pop("models", ["Xray", "NMR"])
        if isinstance(models, str):
            models = [m.strip() for m in models.split(",") if m.strip()]
        kwargs["models"] = [m for m in models if m in MODEL_NAMES]
        return cls(**kwargs)

    def effective_steps(self) -> int:
        """Schedule points setting; two points produce one model forward."""
        return 2 if self.single_step else self.steps


@dataclass
class SampleConfig:
    name: str
    msa_path: str
    fasta_path: Optional[str] = None
    seed: Optional[int] = None
    init_pdb: Optional[str] = None


@dataclass
class UserConfig:
    output_dir: str
    samples: List[SampleConfig] = field(default_factory=list)
    options: RunOptions = field(default_factory=RunOptions)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserConfig":
        return cls(
            output_dir=d.get("output_dir", ""),
            samples=[SampleConfig(**s) for s in d.get("samples", [])],
            options=RunOptions.from_dict(d.get("options")),
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def default_env_path() -> Path:
    return REPO / "config" / "env_config.json"


def load_env_config(path: Optional[str] = None) -> EnvConfig:
    p = Path(path) if path else default_env_path()
    if not p.exists():
        raise FileNotFoundError(
            f"environment config not found: {p}\n"
            f"Copy config/env_config.json from the template and fill in local paths."
        )
    with open(p) as f:
        data = json.load(f)
    return EnvConfig.from_dict(data).resolve()


def load_user_config(path: str) -> UserConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input config not found: {p}")
    with open(p) as f:
        data = json.load(f)
    return UserConfig.from_dict(data)


def env_to_dict(env: EnvConfig) -> Dict[str, Any]:
    """Serializable (json-safe) snapshot of the env config for input.json."""
    return {
        "trflow_xray": env.trflow_xray,
        "trflow_nmr": env.trflow_nmr,
        "esm_weights": env.esm_weights,
        "openfold": {
            "python": env.openfold.python,
            "runner": env.openfold.runner,
            "model_name": env.openfold.model_name,
            "param_path": env.openfold.param_path,
            "timeout_seconds": env.openfold.timeout_seconds,
            "cpus": env.openfold.cpus,
        },
        "gpus": env.gpus,
    }


def asdict_safe(obj) -> Dict[str, Any]:
    """dataclasses.asdict() with non-serializable fields dropped."""
    return {k: v for k, v in asdict(obj).items() if v is not None}

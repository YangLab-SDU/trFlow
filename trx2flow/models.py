"""Model construction, weight loading and process-level caches.

``RepresentationNetwork`` and ``StructureFlowNetwork`` are built from the shared
``MODEL_CONFIG`` and populated from the *same* checkpoint (a merged state dict)
via ``load2_weights`` (weights_only=True, strict=False). ESM weights (1.3GB) and
the trX2 checkpoints are cached at process level so a batch run loads each only
once.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

import esm
from .networks import MODEL_CONFIG, RepresentationNetwork, StructureFlowNetwork

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    model_name: str
    repr_model: nn.Module
    structure_model: nn.Module
    device: torch.device


_MODEL_CACHE: dict = {}
_ESM_CACHE: dict = {}


def load2_weights(model1: nn.Module, model2: nn.Module, weight_file: str, device) -> Tuple[nn.Module, nn.Module]:
    """Load a merged checkpoint into both subnets; unmatched keys are logged, not fatal."""
    weights = torch.load(weight_file, map_location=device, weights_only=True)
    missing1, unexpected1 = model1.load_state_dict(weights, strict=False)
    missing2, unexpected2 = model2.load_state_dict(weights, strict=False)
    if missing1 or missing2:
        logger.warning(
            "%s: %d missing / %d unexpected weights in repr net, "
            "%d missing / %d unexpected in structure net",
            weight_file, len(missing1), len(unexpected1), len(missing2), len(unexpected2),
        )
    return model1.to(device), model2.to(device)


def build_models(model_name: str, ckpt_path: str, device) -> LoadedModels:
    """Construct the representation + structure networks and load the checkpoint."""
    repr_model = RepresentationNetwork(dim_2d=128, dropout=0.1, depth_2d=12, config=MODEL_CONFIG)
    structure_model = StructureFlowNetwork(
        dim_2d=128, dim_3d=128, dim_time=256, depth_xt=4, config=MODEL_CONFIG
    )
    repr_model, structure_model = load2_weights(repr_model, structure_model, ckpt_path, device)
    repr_model.eval()
    structure_model.eval()
    return LoadedModels(model_name=model_name, repr_model=repr_model, structure_model=structure_model, device=device)


def get_model(model_name: str, ckpt_path: str, device) -> LoadedModels:
    """Cached model pair keyed by (model_name, device)."""
    key = (model_name, str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        cached = build_models(model_name, ckpt_path, device)
        _MODEL_CACHE[key] = cached
    return cached


def get_esm_model(esm_path: str, device) -> Tuple[nn.Module, object]:
    """Cached (ESM model, alphabet) pair; model is eval() and grad-free."""
    cached = _ESM_CACHE.get(str(device))
    if cached is None:
        # The checkpoint intentionally omits contact-regression weights. This
        # pipeline uses layer-12 representations and row attention only, so the
        # contact-head warnings are expected and would only clutter the CLI.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*regression weights.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*Regression weights not found.*", category=UserWarning)
            esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_path)
        esm_model.eval()
        esm_model = esm_model.to(device)
        for param in esm_model.parameters():
            param.requires_grad = False
        cached = (esm_model, alphabet)
        _ESM_CACHE[str(device)] = cached
    return cached

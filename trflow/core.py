"""Core flow-matching inference primitives.

This module owns the MSA/repr machinery (``parse_a3m``, ``aa_to_index``,
``mymsa_to_esmmsa``), the ESM forward pass (``esm_forward``), the structure
data container ``Inf_StruData`` and the main ``FlowInferenceCore`` (repr
extraction, structure updates, single-step sampling). Key design points:

  * npz output is handled by ``trflow.output``; ``get_repr`` returns tensors
    only.
  * ESM weights are loaded once and cached (``trflow.models._ESM_CACHE``)
    instead of being reloaded on every ``get_repr`` call.
  * ``flow_mode`` is fixed to ``'iterative'`` (``dist_to_39`` encoding).
  * ``single_step`` uses one model forward. Its schedule ``[1, s, 0]`` retains a
    noisy intermediate state while ``schedule[1:]`` yields exactly one
    structure-model forward.
"""
from __future__ import annotations

import os
import random
import string
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .flow_math import HarmonicPrior, pseudo_beta_fn, rmsdalign
from .geometry import dist_to_39, pseudo_beta_fn_from_pdb
from .trRosettaX2.strutils.utils_3d.prot_converter import ProtConverter

# ---------------------------------------------------------------------------
# Data helpers (ported verbatim)
# ---------------------------------------------------------------------------

_ESM_TOKENS = [5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8, 11, 22, 19, 7, 30, 32]


def mymsa_to_esmmsa(msa, input_type="msa", in_torch=False):
    """Map the 21-letter alphabet to ESM MSA token ids (1-gram)."""
    if in_torch:
        device = msa.device
        token = torch.tensor(_ESM_TOKENS, device=device)
        cls = torch.zeros_like(msa[..., 0:1], device=device)
        eos = 2 * torch.ones_like(msa[..., 0:1], device=device)
        if input_type == "fasta":
            return torch.cat([cls, token[msa], eos], dim=-1)
        return torch.cat([cls, token[msa]], dim=-1)
    token = np.array(_ESM_TOKENS)
    cls = np.zeros_like(msa[..., 0:1])
    eos = 2 * np.ones_like(msa[..., 0:1])
    if input_type == "fasta":
        return np.concatenate([cls, token[msa], eos], axis=-1)
    return np.concatenate([cls, token[msa]], axis=-1)


def parse_a3m(filename, limit=20000):
    """Parse A3M records into an int array of shape [C, L].

    Lowercase insertions are removed. Wrapped sequence records and blank lines
    are accepted; records whose aligned length differs from the query are
    skipped, preserving the behavior of the original inference code.
    """
    records = []
    current = None
    with open(filename, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records.append("".join(current))
                current = []
            elif current is None:
                raise ValueError(f"A3M sequence appears before its first header: {filename}")
            else:
                current.append(line)
        if current is not None:
            records.append("".join(current))

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase + "*"))
    aligned = [record.translate(table) for record in records]
    if not aligned:
        return np.empty((0, 0), dtype=np.uint8)
    query_len = len(aligned[0])
    seqs = [sequence for sequence in aligned if len(sequence) == query_len][:limit]

    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype="|S1").view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    msa[msa > 20] = 20  # treat unknown characters as gaps
    return msa


aa_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "X"]
aa_to_index = {aa: idx for idx, aa in enumerate(aa_list)}


def _onehot_argmax(raw_seq: str) -> np.ndarray:
    aatype = np.zeros((len(raw_seq), 21))
    for i, aa in enumerate(raw_seq):
        aatype[i, aa_to_index.get(aa, 20)] = 1
    return np.argmax(aatype, axis=-1)


def smooth_steps(n, alpha=5):
    """Dirichlet-distributed step sizes (sum=1, ``alpha`` larger = more uniform)."""
    steps = np.random.dirichlet([alpha] * n)
    points = np.cumsum(steps)
    return np.insert(points, 0, 0)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def output_to_pdb(name, raw_seq, model_output, out_dir):
    """Export a structure-model output to ``{out_dir}/{name}.pdb``; returns (path, mean plddt)."""
    os.makedirs(out_dir, exist_ok=True)
    plddt = model_output["plddt"][-1].squeeze().cpu().numpy()
    unrelaxed_model = os.path.join(out_dir, f"{name}.pdb")
    cords_prot = model_output["cords_allatm"][-1].squeeze(0).permute(1, 0, 2)
    lines_prot = ProtConverter.export_pdb_file(
        raw_seq, cords_prot.squeeze(0).data.cpu().numpy(), path=None, chain_id="A", ca_only=False, confidence=plddt,
    )
    with open(unrelaxed_model, "w") as f:
        f.write("".join(lines_prot))
    return unrelaxed_model, float(plddt.mean())


def esm_forward(esm_model, msa_tokens, res_id, device):
    """Run ESM-MSA-1b on ``msa_tokens`` (tensor [1, C, L+1]); returns (repr12, row_attn)."""
    with torch.no_grad():
        with torch.amp.autocast(device_type=str(device).split(":")[0]):
            emb_out = esm_model(
                msa_tokens,
                repr_layers=[12],
                need_head_weights=True,
                res_idx=res_id,
            )
            emb_repr = emb_out["representations"][12].detach().contiguous()
            row_attn = emb_out["row_attentions"].detach().contiguous()
            del emb_out
            torch.cuda.empty_cache()
    return emb_repr, row_attn


# ---------------------------------------------------------------------------
# Struct data
# ---------------------------------------------------------------------------

@dataclass
class Inf_StruData:
    raw_seq: str
    reprs: dict
    pred_gemo: dict
    cb_mask: torch.Tensor
    aatype: torch.Tensor
    schedule: Optional[np.ndarray] = None
    t_step: Optional[torch.Tensor] = None
    noisy: Optional[torch.Tensor] = None
    noisy_cb_dist: Optional[torch.Tensor] = None
    noisy_cb: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

class FlowInferenceCore:
    """One (repr + structure) model pair on one device, plus the cached ESM model."""

    def __init__(self, model_name: str, ckpt_path: str, esm_path: str, device):
        from trflow.models import get_esm_model, get_model

        self.model_name = model_name
        self.device = device
        self.models = get_model(model_name, ckpt_path, device)
        self.repr_model = self.models.repr_model
        self.structure_model = self.models.structure_model
        self.esm_model, _alphabet = get_esm_model(esm_path, device)

    # -- Stage 1: sequence representation -----------------------------------
    def get_repr(self, raw_seq: str, msa_path: str) -> dict:
        """Run the repr network; returns ``mid_data`` {pred_gemo, reprs, raw_seq, name}."""
        self.repr_model.eval()
        device = self.device
        L = len(raw_seq)
        msa = parse_a3m(msa_path)[None]  # [1, C, L]
        msa = torch.from_numpy(msa).to(device)
        res_id = torch.arange(L).view(1, L).to(device)

        msa_esm = mymsa_to_esmmsa(msa.long(), input_type="msa", in_torch=True).to(device)
        emb_repr, row_attn = esm_forward(self.esm_model, msa_esm[:, :350], res_id, device=device)
        emb_out = {"representations": {12: emb_repr}, "row_attentions": row_attn}
        with torch.no_grad():
            pred = self.repr_model(
                raw_seq=raw_seq, msa=msa_esm, emb_out=emb_out, res_id=res_id, device=device, msa_cutoff=350,
            )
        reprs = pred["repr"]
        pred_2d = pred["pred_gemo"]
        return {
            "pred_gemo": {
                "dist": pred_2d["dist"][0],
                "theta": pred_2d["theta"][0],
                "omega": pred_2d["omega"][0],
                "phi": pred_2d["phi"][0],
            },
            "reprs": {"pair": reprs["pair"], "single": reprs["single"], "msa": reprs["msa"]},
            "raw_seq": raw_seq,
            "name": self.model_name,
        }

    # -- StruData construction ----------------------------------------------
    def build_inf_strudata(self, mid_data: dict):
        """Construct ``(Inf_StruData, aatype, HarmonicPrior)`` from a ``get_repr`` result."""
        raw_seq = mid_data["raw_seq"]
        aatype = torch.from_numpy(_onehot_argmax(raw_seq)).to(self.device)
        prior = HarmonicPrior(len(raw_seq))
        prior.to(self.device)
        inf_strudata = Inf_StruData(
            raw_seq=raw_seq,
            reprs=mid_data,
            pred_gemo={k: (v[None] if v.ndim == 3 else v) for k, v in mid_data["pred_gemo"].items()},
            cb_mask=torch.ones(1, len(raw_seq), device=self.device),
            aatype=aatype,
        )
        return inf_strudata, aatype, prior

    def get_init_pseudo_beta(self, init_pdb: str) -> torch.Tensor:
        """CB pseudo-atom coordinates read from an init structure PDB."""
        pseudo_beta = pseudo_beta_fn_from_pdb(init_pdb, None)
        return pseudo_beta.to(self.device)

    # -- Sampling -----------------------------------------------------------
    @staticmethod
    def make_schedule(steps: int, random_step_size: bool) -> np.ndarray:
        """Build the noise schedule for a configured number of flow steps.

        ``steps`` is the user-facing number of structure-model forwards. The
        full schedule therefore has ``steps + 2`` points: the initial noise
        endpoint, one starting point per forward, and the final zero endpoint.
        The initial endpoint is removed before ``get_stru_repr`` iterates over
        adjacent pairs.
        """
        if steps < 1:
            raise ValueError("model forwards must be at least 1")
        schedule_intervals = steps + 1
        if random_step_size:
            return smooth_steps(schedule_intervals, alpha=5)[::-1]
        return np.linspace(1.0, 0.0, schedule_intervals + 1)

    def update_inf_strudata(self, inf_strudata, pseudo_beta, prior, steps, random_step, random_step_size):
        """Inject noise toward ``pseudo_beta`` and set the sampling schedule."""
        tmax = 1.0
        if random_step:
            steps = random.choice([1, 7])
        schedule = self.make_schedule(steps, random_step_size)
        if tmax != 1.0:
            schedule = np.array([1.0] + list(schedule))

        t, s = schedule[:-1][0], schedule[1:][0]
        if pseudo_beta is not None:
            noisy = prior.sample()
            noisy = rmsdalign(pseudo_beta, noisy)
            noisy_cb = (s / t) * noisy + (1 - s / t) * pseudo_beta
            inf_strudata.noisy = noisy
            inf_strudata.noisy_cb = noisy_cb
            inf_strudata.noisy_cb_dist = dist_to_39(
                torch.sum((noisy_cb[None].unsqueeze(-2) - noisy_cb[None].unsqueeze(-3)) ** 2, dim=-1) ** 0.5
            )
        inf_strudata.t_step = torch.ones(1, device=self.device) * s
        inf_strudata.schedule = schedule[1:]
        return inf_strudata

    def get_stru_repr(self, inf_strudata) -> List[dict]:
        """Run the structure/flow model over the schedule; returns per-step outputs."""
        self.structure_model.eval()
        device = self.device
        schedule = inf_strudata.schedule
        noisy_cb = inf_strudata.noisy_cb
        data_update = {
            "noisy_cb": noisy_cb,
            "noisy_cb_dist": inf_strudata.noisy_cb_dist,
            "t_step": inf_strudata.t_step,
        }
        outputs = []
        with torch.no_grad():
            for t, s in zip(schedule[:-1], schedule[1:]):
                output = self.structure_model(
                    raw_seq=inf_strudata.raw_seq,
                    reprs=inf_strudata.reprs["reprs"],
                    _reprs=inf_strudata.reprs["reprs"],
                    pred_gemos=inf_strudata.pred_gemo,
                    t_step=data_update["t_step"],
                    noisy_cb_dist=data_update["noisy_cb_dist"],
                    cb_mask=inf_strudata.cb_mask,
                    noisy_cb=data_update["noisy_cb"],
                    device=device,
                )
                pseudo_beta = pseudo_beta_fn(
                    inf_strudata.aatype, output["cords_allatm"][-1].squeeze(0).permute(1, 0, 2), None
                )
                outputs.append(output)
                noisy_cb = rmsdalign(pseudo_beta, noisy_cb)
                noisy_cb = (s / t) * noisy_cb + (1 - s / t) * pseudo_beta
                data_update["noisy_cb"] = noisy_cb
                data_update["noisy_cb_dist"] = dist_to_39(
                    torch.sum((noisy_cb[None].unsqueeze(-2) - noisy_cb[None].unsqueeze(-3)) ** 2, dim=-1) ** 0.5
                )
                data_update["t_step"] = torch.ones(1, device=noisy_cb.device) * s
        return outputs

    def sample_one(self, inf_strudata, pseudo_beta, prior, out_dir, name, steps, random_step, random_step_size):
        """One full sample: inject noise, denoise, export PDB. Returns (path, mean plddt)."""
        inf_strudata = self.update_inf_strudata(
            inf_strudata, pseudo_beta, prior, steps=steps, random_step=random_step, random_step_size=random_step_size,
        )
        outputs = self.get_stru_repr(inf_strudata)
        return output_to_pdb(name=name, raw_seq=inf_strudata.raw_seq, model_output=outputs[-1], out_dir=out_dir)

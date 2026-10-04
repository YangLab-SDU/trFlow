"""Geometric exploration (Stage 2).

``blend_distogram`` combines the current structural CB-CB distance matrix with
the model's predicted distogram; ``run_exploration`` is the surrounding loop
that feeds the blended distogram back into the structure model.

Exploration repeatedly (a) runs the structure model guided only by a blended
distogram and (b) re-blends that distogram with the structural CB-CB distances,
until the distogram stops changing (``|Δ| < 0.01``) or ``max_iters`` is hit.
The converged structure becomes the base for the "updated" sampling paths.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .flow_math import pseudo_beta_fn
from .geometry import convert_dgram_37_to_39, dist_to_37, process_distribution_fast

from .core import FlowInferenceCore, Inf_StruData


def get_iter_pseudo_beta(core: FlowInferenceCore, inf_strudata: Inf_StruData, new_dist_39: torch.Tensor):
    """Structure-model forward guided only by ``new_dist_39`` (no initial coords)."""
    device = core.device
    with torch.no_grad():
        iter_output = core.structure_model(
            raw_seq=inf_strudata.raw_seq,
            reprs=inf_strudata.reprs["reprs"],
            _reprs=inf_strudata.reprs["reprs"],
            pred_gemos=inf_strudata.pred_gemo,
            t_step=torch.tensor(
                [FlowInferenceCore.make_schedule(1, random_step_size=True)[1]],
                device=device,
                dtype=torch.float32,
            ),
            noisy_cb_dist=new_dist_39,
            cb_mask=inf_strudata.cb_mask,
            noisy_cb=None,
            device=device,
        )
    pseudo_beta = pseudo_beta_fn(
        inf_strudata.aatype, iter_output["cords_allatm"][-1].squeeze(0).permute(1, 0, 2), None
    )
    return pseudo_beta, iter_output


def blend_distogram(
    core: FlowInferenceCore,
    inf_strudata: Inf_StruData,
    model_output: dict,
    iter_dist_37: torch.Tensor,
    iter_tmp_dist_37: np.ndarray,
):
    """Blend the structural CB-CB distogram into the running distogram.

    ``iter_dist_37`` / ``iter_tmp_dist_37`` are the unbatched [L, L, 37]
    distograms (``reprs["pred_gemo"]["dist"]``). Returns the updated
    ``(iter_dist_37, iter_tmp_dist_37, new_dist_39, iter_flag)`` where
    ``iter_flag=False`` means the distogram converged.
    """
    device = core.device
    old_pseudo_beta = pseudo_beta_fn(
        inf_strudata.aatype, model_output["cords_allatm"][-1].squeeze(0).permute(1, 0, 2), None
    )
    old_fact_dist_37 = dist_to_37(
        torch.sum((old_pseudo_beta[None].unsqueeze(-2) - old_pseudo_beta[None].unsqueeze(-3)) ** 2, dim=-1) ** 0.5,
        device=device,
    )
    new_dist_37 = process_distribution_fast(iter_dist_37.cpu().numpy(), old_fact_dist_37.cpu().numpy()[0])
    new_dist_37 = torch.from_numpy(new_dist_37).to(device)
    tmp_new_dist_37 = process_distribution_fast(iter_tmp_dist_37, old_fact_dist_37.cpu().numpy()[0], norm=False)
    if np.max(np.abs(iter_tmp_dist_37 - tmp_new_dist_37)) < 0.01:
        iter_flag = False
    else:
        iter_tmp_dist_37 = tmp_new_dist_37
        iter_flag = True
    iter_dist_37 = new_dist_37
    new_dist_39 = convert_dgram_37_to_39(new_dist_37[None])
    return iter_dist_37, iter_tmp_dist_37, new_dist_39, iter_flag


def run_exploration(
    core: FlowInferenceCore,
    inf_strudata: Inf_StruData,
    init_pseudo_beta: torch.Tensor,
    prior,
    max_iters: int = 1000,
) -> Tuple[torch.Tensor, dict, int, bool]:
    """Run the exploration loop until the distogram converges.

    Returns ``(updated_pseudo_beta, iter_output, n_iters, converged)`` where
    ``iter_output`` is the final structure-model output (for PDB export).
    """
    device = core.device
    pseudo_beta = init_pseudo_beta.clone()
    iter_dist_37 = inf_strudata.reprs["pred_gemo"]["dist"]  # [L, L, 37] (unbatched)
    iter_tmp_dist_37 = iter_dist_37.cpu().numpy()
    iter_output = None
    converged = False
    new_dist_39 = None  # produced by blend_distogram; consumed on iter_num >= 2
    for iter_num in range(1, max_iters + 1):
        if iter_num == 1:
            # Seed structure: one denoise from the init structure (mirrors iteration 0
            # of the original loop). One model forward is used and its
            # intermediate time is sampled just as in flow_inferenceV3.py.
            inf_strudata = core.update_inf_strudata(
                inf_strudata, pseudo_beta, prior, steps=1, random_step=False, random_step_size=True,
            )
            iter_output = core.get_stru_repr(inf_strudata)[-1]
        else:
            pseudo_beta, iter_output = get_iter_pseudo_beta(core, inf_strudata, new_dist_39)
        iter_dist_37, iter_tmp_dist_37, new_dist_39, iter_flag = blend_distogram(
            core, inf_strudata, iter_output, iter_dist_37, iter_tmp_dist_37
        )
        if not iter_flag:
            converged = True
            break
    updated_pseudo_beta = pseudo_beta_fn(
        inf_strudata.aatype, iter_output["cords_allatm"][-1].squeeze(0).permute(1, 0, 2), None
    ).to(device)
    return updated_pseudo_beta, iter_output, iter_num, converged

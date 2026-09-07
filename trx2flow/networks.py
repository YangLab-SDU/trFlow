"""Representation network and structure/flow network with shared model config."""
from __future__ import annotations

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .trRosettaX2.evoutils.attn_conv_e2e import Predictor2D
from .trRosettaX2.strutils.structure_module import StructureModuleFullAtom
from .trRosettaX2.strutils.utils_3d.rigid_utils import Rigid, Rotation
from .time_embedding import Derf, FlowFormer, GaussianFourierProjection, TimestepEmbedder

MODEL_CONFIG = {
    "structure_module": {
        "c_s": 128,
        "c_z": 128,
        "c_ipa": 16,
        "c_resnet": 128,
        "no_heads_ipa": 12,
        "no_qk_points": 4,
        "no_v_points": 8,
        "no_blocks": 8,
        "no_transition_layers": 1,
        "no_resnet_blocks": 2,
        "no_angles": 5,
        "trans_scale_factor": 10.0,
        "refinenet": {"enable": False, "dim": 64, "is_pos_emb": True, "n_layer": 4},
    },
    "use_noisy_rigids": False,
    "use_derf": True,
    "use_esm": True,
}


def one_hot(x, bin_values=torch.arange(2, 20.5, .5)):
    bin_values = bin_values.to(x.device)
    n_bins = len(bin_values)
    bin_values = bin_values.view([1] * x.ndim + [-1])
    binned = (bin_values <= x[..., None]).sum(-1)
    binned = torch.where(binned > n_bins - 1, n_bins - 1, binned)
    onehot = (torch.arange(n_bins, device=x.device) == binned[..., None]).float()
    return onehot


class InputEmbedder(nn.Module):
    """2D/1D/MSA features from an ESM-MSA embedding (used when ``use_esm``)."""

    def __init__(self):
        super(InputEmbedder, self).__init__()

    def forward(self, msa, emb_out=None):
        with torch.no_grad():
            f2d, msa_emb = self.get_f2d(msa, emb_out=emb_out)
        return {'pair': f2d, 'msa': msa_emb}

    def get_f2d(self, msa, emb_out):
        device = msa.device
        nrow, ncol = msa.shape[-2], msa.shape[-1] - 1
        emb_repr = emb_out['representations'][12][:, :, 1:]
        seq_emb = emb_repr[0, 0]
        row_attn = emb_out['row_attentions'][:, :, :, 1:, 1:]
        seq_emb = torch.cat([
            seq_emb[None, ...].repeat(ncol, 1, 1),
            seq_emb[:, None, ...].repeat(1, ncol, 1),
        ], dim=-1)[None, ...]  # 1, L, L, 2*768
        row_attn = Rearrange('b l h m n -> b m n (l h)')(row_attn)  # 1, L, L, 144

        msa1hot = (torch.arange(31, device=device) == msa[0, :, 1:, None]).float()
        w = self.reweight(msa1hot, .8)

        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 962], device=device)

        f2d = f2d_dca.view([1, ncol, ncol, 962])
        return torch.cat([seq_emb, row_attn, f2d], dim=-1), emb_repr

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(axis=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(axis=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(axis=-1)
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        device = msa1hot.device
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())
        mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)
        nc_eye = torch.eye(nc, device=device)
        x3 = torch.sqrt(torch.square(x1[:, :-1, :, :-1]).sum((1, 3))) * (1 - nc_eye)
        apc = x3.sum(axis=0, keepdims=True) * x3.sum(axis=1, keepdims=True) / x3.sum()
        contacts = (x3 - apc) * (1 - nc_eye)

        return torch.cat([features, contacts[:, :, None]], dim=2)


class noESMInputEmbedder(nn.Module):
    """Sequence/PSSM/DCA features computed directly from the MSA (no ESM)."""

    def __init__(self):
        super(noESMInputEmbedder, self).__init__()

    def forward(self, msa, emb_out=None):
        with torch.no_grad():
            f2d, msa_emb = self.get_f2d(msa[0])
        return {'pair': f2d, 'msa': msa_emb}

    def get_f2d(self, msa):
        device = msa.device
        nrow, ncol = msa.size()[-2:]

        msa1hot = (torch.arange(21, device=device) == msa[..., None]).float()
        w = self.reweight(msa1hot, .8)

        f1d_seq = msa1hot[0, :, :20]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 442], device=device)

        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                        f1d[None, :, :].repeat([ncol, 1, 1]),
                        f2d_dca], dim=-1)
        f2d = f2d.view([1, ncol, ncol, 442 + 2 * 42])
        return f2d, None

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(axis=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(axis=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(axis=-1)
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        device = msa1hot.device
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())
        mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)
        nc_eye = torch.eye(nc, device=device)
        x3 = torch.sqrt(torch.square(x1[:, :-1, :, :-1]).sum((1, 3))) * (1 - nc_eye)
        apc = x3.sum(axis=0, keepdims=True) * x3.sum(axis=1, keepdims=True) / x3.sum()
        contacts = (x3 - apc) * (1 - nc_eye)

        return torch.cat([features, contacts[:, :, None]], dim=2)


class RecyclingEmbedder(nn.Module):
    def __init__(self, dim=128, use_derf=False):
        super(RecyclingEmbedder, self).__init__()
        self.linear = nn.Linear(37, dim)
        if use_derf:
            self.norm_pair = Derf(dim)
            self.norm_msa = Derf(dim)
        else:
            self.norm_pair = nn.LayerNorm(dim)
            self.norm_msa = nn.LayerNorm(dim)

    def forward(self, reprs_prev, x=None):
        if x is None:
            x = reprs_prev['x']
        d = torch.cdist(x, x)
        d = one_hot(d)
        d = self.linear(d)
        pair = self.norm_pair(reprs_prev['pair']) + d
        single = self.norm_msa(reprs_prev['single'])
        return single, pair


class RepresentationNetwork(nn.Module):
    """Transformer over the MSA producing pair/single/MSA representations and
    predicted geometries (distogram + orientation angles)."""

    def __init__(self, dim_2d=128, dropout=.1, depth_2d=12, config=MODEL_CONFIG):
        super(RepresentationNetwork, self).__init__()
        self.config = config
        use_derf = self.config['use_derf']
        if self.config['use_esm']:
            self.input_embedder = InputEmbedder()
            self.net2d = Predictor2D(in_dim=1680 + 962, dim=128, depth=depth_2d, num_tokens=31, msa_tie_row_attn=True,
                                    attn_dropout=dropout, ff_dropout=dropout, use_derf=use_derf)
        else:
            self.input_embedder = noESMInputEmbedder()
            self.net2d = Predictor2D(in_dim=526, dim=128, depth=depth_2d, num_tokens=21, msa_tie_row_attn=True,
                                    attn_dropout=dropout, ff_dropout=dropout, use_derf=use_derf)
        self.recycle_embedder = RecyclingEmbedder(dim=dim_2d, use_derf=use_derf)

    def forward(self, raw_seq, msa, msa_filtered=None, emb_out=None, res_id=None,
                device='cuda:0', msa_cutoff=500):
        reprs_prev = None
        L = len(raw_seq)
        if msa_filtered is None:
            msa_filtered = msa

        feats = self.input_embedder(msa, emb_out=emb_out)
        if reprs_prev is None:
            reprs_prev = {
                'pair': torch.zeros((1, L, L, 128), device=device),
                'single': torch.zeros((1, L, 128), device=device),
                'x': torch.zeros((1, L, 3), device=device),
            }
            t = reprs_prev['x']

        rec_single, rec_pair = self.recycle_embedder(reprs_prev, t)
        rec_reprs = {'single': rec_single, 'pair': rec_pair}

        if self.config['use_esm']:
            pred_gemos, reprs = self.net2d(f2d=feats['pair'], msa=msa_filtered[:, :msa_cutoff, 1:],
                                            msa_emb=feats['msa'][:, :msa_cutoff, :],
                                            rec_reprs=rec_reprs,
                                            res_id=res_id, preprocess=True, to_prob=True, return_repr=True)
        else:
            pred_gemos, reprs = self.net2d(f2d=feats['pair'], msa=msa_filtered[:, :msa_cutoff, :], msa_emb=None,
                                            rec_reprs=rec_reprs,
                                            res_id=res_id, preprocess=True, to_prob=True, return_repr=True)
        reprs['pair'] = rearrange(reprs['pair'], 'b d i j -> b i j d')

        _reprs = {"single": reprs['msa'][:, 0], "pair": reprs['pair'], 'msa': reprs['msa']}
        return {'repr': _reprs, 'pred_gemo': pred_gemos}


class StructureFlowNetwork(nn.Module):
    """Denoising structure model: mixes noisy Cb-distograms into the semantic
    pair representation and predicts the structure + a 37-bin distogram."""

    def __init__(self, dim_2d=128, dim_3d=128, dim_time=256, depth_xt=12, config=MODEL_CONFIG):
        super(StructureFlowNetwork, self).__init__()
        self.config = config
        use_derf = self.config['use_derf']
        config['structure_module']['use_derf'] = use_derf

        self.input_time_embedding = nn.Linear(dim_time, 128)
        self.time_step_embedding = TimestepEmbedder(dim=256, nfreq=256)
        # Keep the attribute name for released-checkpoint compatibility; the
        # implementation is the FlowFormer pair/MSA fusion module.
        self.input_pair_stack = FlowFormer(dim=dim_2d, no_blocks=depth_xt, use_derf=use_derf)
        self.input_pair_embedding = nn.Linear(39, dim_2d)
        self.xt_time_step_embedding = GaussianFourierProjection(embedding_size=256)
        self.to_dist_logits = nn.Conv2d(dim_2d, 37, 1)

        self.structure_module = StructureModuleFullAtom(
            **config['structure_module']
        )
        if use_derf:
            self.to_plddt = nn.Sequential(
                Derf(dim_3d),
                nn.Linear(dim_3d, dim_3d),
                nn.ReLU(),
                nn.Linear(dim_3d, dim_3d),
                nn.ReLU(),
                nn.Linear(dim_3d, 50),
            )
        else:
            self.to_plddt = nn.Sequential(
                nn.LayerNorm(dim_3d),
                nn.Linear(dim_3d, dim_3d),
                nn.ReLU(),
                nn.Linear(dim_3d, dim_3d),
                nn.ReLU(),
                nn.Linear(dim_3d, 50),
            )

    def _get_input_pair_embeddings(self, dgram, mask, t_step, pair, m):
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
        inp_z, m_updated = self.input_pair_stack(inp_z, m, pair)

        t_embed = self.input_time_embedding(self.xt_time_step_embedding(t_step))[None, None]
        inp_z = inp_z + t_embed
        x = inp_z.permute(0, 3, 1, 2)

        trunk_embeds = (x + rearrange(x, 'b d i j -> b d j i')) * 0.5  # symmetrize
        dist_logits = self.to_dist_logits(trunk_embeds)
        return inp_z, dist_logits.permute(0, 2, 3, 1).softmax(-1), m_updated

    def forward(self, raw_seq, reprs, _reprs, pred_gemos, t_step, cb_mask=None, noisy_cb_dist=None, noisy_cb=None,
                device='cuda:0'):
        L = len(raw_seq)

        if self.config['use_noisy_rigids'] and noisy_cb is not None:
            rigids = Rigid(
                Rotation.identity(noisy_cb.shape[:-1], noisy_cb.dtype, device, self.training, fmt='quat'),
                noisy_cb / 10,
            )
        else:
            rigids = None

        if noisy_cb_dist is not None:
            if t_step is None:
                t_step = _reprs['pair'].new_zeros(1)
            cb_embed, xt_dist_logits, m_updated = self._get_input_pair_embeddings(
                noisy_cb_dist, cb_mask, t_step=t_step, pair=_reprs['pair'], m=reprs['msa']
            )
            t_embed = self.input_time_embedding(self.time_step_embedding(t_step))[None]
        else:
            cb_embed, xt_dist_logits, m_updated = self._get_input_pair_embeddings(
                _reprs['pair'].new_zeros(1, L, L, 39), _reprs['pair'].new_ones(1, L),
                t_step=_reprs['pair'].new_zeros(1), pair=_reprs['pair'], m=_reprs['msa'],
            )
            t_embed = self.input_time_embedding(self.time_step_embedding(_reprs['pair'].new_zeros(1)))[None]

        t_embed_reprs = {}
        t_embed_reprs['pair'] = cb_embed
        t_embed_reprs['single'] = m_updated[:, 0] + t_embed
        outputs = self.structure_module.forward(raw_seq, t_embed_reprs, rigids=rigids, return_mid=True)

        rots = []
        tsls = []
        for frames in outputs['scaled_frames']:
            rigid = Rigid.from_tensor_7(frames, normalize_quats=True)
            fram = rigid.to_tensor_4x4()
            rots.append(fram[:, :, :3, :3])
            tsls.append(fram[:, :, :3, 3:].squeeze(-1))
        outputs['frames'] = (torch.stack(rots, dim=0), torch.stack(tsls, dim=0))
        outputs['geoms'] = pred_gemos

        plddt_prob = self.to_plddt(outputs['single'][-1]).softmax(-1)
        plddt = torch.einsum('bik,k->bi', plddt_prob, torch.arange(0.01, 1.01, 0.02, device=device))
        outputs['plddt_prob'] = plddt_prob
        outputs['plddt'] = plddt
        outputs['xt_dist_logits'] = xt_dist_logits

        return outputs

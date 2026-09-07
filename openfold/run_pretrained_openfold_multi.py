# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from the OpenFold project (Apache-2.0); a multi-chain prediction
# variant of run_pretrained_openfold.py. Modified for this repo: sys.path and
# default paths are neutralized, helper imports renamed to utils_common.

import argparse
from datetime import date
import logging
from pathlib import Path

import numpy as np
import os

import pickle
import random
import sys
import time
import torch
from tqdm import tqdm
import sys

# Path-neutralized: was a hardcoded absolute-path sys.path hack. Resolve
# against this file's own directory so the repo is self-contained.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants, protein
# import openfold.np.relax.relax as relax
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

from scripts.utils import add_data_args
from utils_common.deeplearn import get_UUID, find_empty_gpu


def main(args):
    # Feature processing (MSA masking / sampling in data_transforms.py) draws on
    # the global torch/np RNG; the vendored script originally drew a fresh random
    # seed each run, making outputs unreproducible. Seed it deterministically.
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = 42
    random_seed = int(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    sample_lst = open(args.lst).read().splitlines()
    if not args.recalc:
        if args.save_mid:
            sample_lst = [pid for pid in sample_lst if not os.path.isfile(os.path.join(
                args.output_dir, pid, f'{args.model_name}_mid.npz'
            ))]
        else:
            sample_lst = [pid for pid in sample_lst if not os.path.isfile(os.path.join(
                args.output_dir, pid, f'{args.model_name}_unrelaxed.pdb'
            ))]

    model = load_model(args)
    config = model_config(args.model_name)
    use_small_bfd = True

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=None,
    )

    output_dir_base = args.output_dir
    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    if args.ppi and args.length_json is not None:
        len_dict = read_json(args.length_json)
    else:
        len_dict = None
    for pid in tqdm(sample_lst):
        fasta = os.path.join(args.fasta_path, f'{pid}.fasta')
        msa = os.path.join(args.msa_path, f'{pid}.a3m')
        if not os.path.exists(msa):
            print(f'miss {msa}')
            continue
        try:
            predict(model, data_processor, feature_processor, pid, fasta, msa, len_dict)
        except RuntimeError as e:
            if 'out of mem' in str(e):
                print(f'{pid} oom!')
            else:
                raise e


def load_model(args):
    config = model_config(args.model_name)
    model = AlphaFold(config)
    # print(sum(p.numel() for p in model.parameters())/1e6)
    # sys.exit(-1)
    import_jax_weights_(model, args.param_path, version=args.model_name)
    # script_preset_(model)
    # model = model.to(args.model_device)
    # if torch.cuda.is_available():
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def predict(model, data_processor, feature_processor, pid, fasta, msa, len_dict=None):
    # Gather input sequences
    with open(fasta, "r") as fp:
        lines = [l.strip() for l in fp.readlines()]

    tags, seqs = lines[::2], lines[1::2]
    tags = [l[1:] for l in tags]

    # for tag, seq in zip(tags, seqs):
    fasta_path = os.path.join(args.output_dir, pid, "tmp.fasta")
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, "w") as fp:
        fp.write(f">{pid}\n{seqs[0]}")
    s = time.time()
    logging.info("Generating features...")
    # if (args.use_precomputed_alignments is None):
    # if not os.path.exists(local_alignment_dir):
    #     os.makedirs(local_alignment_dir)

    # alignment_runner = data_pipeline.AlignmentRunner(
    #     jackhmmer_binary_path=args.jackhmmer_binary_path,
    #     hhblits_binary_path=args.hhblits_binary_path,
    #     hhsearch_binary_path=args.hhsearch_binary_path,
    #     uniref90_database_path=args.uniref90_database_path,
    #     mgnify_database_path=args.mgnify_database_path,
    #     bfd_database_path=args.bfd_database_path,
    #     uniclust30_database_path=args.uniclust30_database_path,
    #     pdb70_database_path=args.pdb70_database_path,
    #     use_small_bfd=True,
    #     no_cpus=args.cpus,
    # )
    # alignment_runner.run(
    #     fasta_path, msa
    # )

    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path, alignment_dir=msa, templates_pdbs=None
    )

    # Remove temporary FASTA file
    os.remove(fasta_path)

    if len_dict is not None:
        res_ids = []
        for n, l in enumerate(len_dict[pid]):
            if n == 0:
                res_ids.append(np.arange(l))
            else:
                res_ids.append(np.arange(len_dict[pid][n - 1], l) + n * 200)
        res_id = np.concatenate(res_ids, axis=0)
        feature_dict['residue_index'] = res_id

    processed_feature_dict = feature_processor.process_features(
        feature_dict, mode='predict',
    )
    t1 = time.time()
    print(f'feats:{t1 - s}')
    logging.info("Executing model...")
    batch = processed_feature_dict
    with torch.no_grad() and torch.cuda.amp.autocast(enabled=False):
        batch = {
            k: torch.as_tensor(v, device=device)
            for k, v in batch.items()
        }

        t = time.perf_counter()
        out = model(batch, save_mid=args.save_mid)
        out['dist'] = out['distogram_logits'].softmax(dim=-1)
        logging.info(f"Inference time: {time.perf_counter() - t}")
        if args.save_mid:
            np.savez_compressed(os.path.join(
                args.output_dir, pid, f'{args.model_name}_mid.npz'
            ), pair=out['pair_mid'].astype(np.float16))
            # pair_mid is only present when save_mid=True; drop it so the
            # downstream PDB export below sees a clean output dict.
            del out['pair_mid']
    t2 = time.time()
    print(f'predict:{t2 - t1}')
    # Toss out the recycling dimensions --- we don't need them anymore
    batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
    out = tensor_tree_map(lambda x: np.array(x.cpu()) if isinstance(x,torch.Tensor) else x, out)
    # print(out.keys())
    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)

    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors
    )

    # Save the unrelaxed PDB.
    unrelaxed_output_path = os.path.join(
        args.output_dir, pid, f'{args.model_name}_unrelaxed.pdb'
    )
    os.makedirs(Path(unrelaxed_output_path).parent, exist_ok=True)
    with open(unrelaxed_output_path, 'w') as f:
        f.write(protein.to_pdb(unrelaxed_protein))
    # print(f'pdb export to {unrelaxed_output_path}')
    t3 = time.time()

    # plddt_out_path = os.path.join(
    #     args.output_dir, pid, f'{tag}_{args.model_name}_plddt.npy'
    # )
    out_npz_path = os.path.join(
        args.output_dir, pid, f'{args.model_name}.npz'
    )
    # np.save(plddt_out_path,plddt)
    save_dict = {
        'dist': out['dist'].astype(np.float16),
        'pair': out['pair'].astype(np.float16),
        'single': out['single'].astype(np.float16),
        'lddt_logits': out['lddt_logits'].astype(np.float16),
        'plddt': out['plddt'].astype(np.float16),
    }
    np.savez_compressed(out_npz_path, **save_dict)


        # save_dict['pair_mid'] = out['pair_mid'].astype(np.float16)
    # print(f'output:{t3 - t2}')
    # amber_relaxer = relax.AmberRelaxation(
    #     use_gpu=(args.model_device != "cpu"),
    #     **config.relax,
    # )

    # # Relax the prediction.
    # t = time.perf_counter()
    # visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    # if ("cuda" in args.model_device):
    #     device_no = args.model_device.split(":")[-1]
    #     os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    # os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    # logging.info(f"Relaxation time: {time.perf_counter() - t}")
    #
    # # Save the relaxed PDB.
    # relaxed_output_path = os.path.join(
    #     args.output_dir, f'{tag}_{args.model_name}_relaxed.pdb'
    # )
    # with open(relaxed_output_path, 'w') as f:
    #     f.write(relaxed_pdb_str)


if __name__ == "__main__":
    home = str(Path.home())
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--lst", type=str, default=f'{home}//db/baker32/32.lst'
    )
    parser.add_argument(
        "--fasta_path", type=str, default=f'{home}/db/baker32/fasta_nolinker/'
    )
    parser.add_argument(
        "--msa_path", type=str,
        # default=f'{home}//db/baker32/msa_concate',
        default=f'{home}//db/baker32/a3m_unpair_nolow',
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str,
        # default=f'{home}/db/baker32/pred/openfold_paired',
        default=f'{home}/db/baker32/pred/openfold_unpair',
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--template_pdb_files", type=str, default=None
    )
    parser.add_argument(
        "--recalc", action='store_true', default=False,
    )
    parser.add_argument(
        "--save_mid", action='store_true', default=False,
    )
    parser.add_argument(
        "--ppi", action='store_true', default=False,
    )
    parser.add_argument(
        "--length_json", type=str,
        # default=f'{home}/db/baker32/cumlen.json',
        default=None,
        help="""only for baker32""",
    )
    parser.add_argument(
        # Neutralized for the GitHub build: the original default was
        # find_empty_gpu(cpu_ok=False), which scanned nvidia-smi at import time
        # and raised if no GPU had >30GB free. Always pass --gpu explicitly.
        "--gpu", type=str, default="0",
        help="""-1:cpu, 0~5:gpu"""
    )
    parser.add_argument(
        "--model_name", type=str, default="model_1_ptm",
        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm, as defined on the AlphaFold GitHub."""
    )
    parser.add_argument(
        "--param_path", type=str, default=None,
        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             openfold/resources/params"""
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--preset', type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        '--data_random_seed', type=str, default=None
    )
    add_data_args(parser)
    args = parser.parse_args()

    if (args.param_path is None):
        # Neutralized: repo-relative guess under models/.
        args.param_path = os.path.join(
            Path(__file__).resolve().parent.parent, "models", "openfold_params_" + args.model_name + ".npz"
        )

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and int(args.gpu) > -1 else torch.device(
        'cpu')
    # if (args.model_device == "cpu" and torch.cuda.is_available()):
    #     logging.warning(
    #         """The model is being run on CPU. Consider specifying
    #         --model_device for better performance"""
    #     )
    main(args)

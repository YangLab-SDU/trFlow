```text
    .            oooooooooooo oooo
  .o8            `888'     `8 `888
.o888oo oooo d8b 888          888   .ooooo.  oooo oooo    ooo
  888   `888""8P 888oooo8     888  d88' `88b  `88. `88.  .8'
  888    888     888    "     888  888   888   `88..]88..8'
  888 .  888     888          888  888   888    `888'`888'
  "888" d888b    o888o        o888o `Y8bod8P'     `8'  `8'
```

# trFlow

Protein conformation sampling with flow matching.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://www.python.org/) [![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c?style=flat-square)](https://pytorch.org/)

## Overview

trFlow generates alternative protein conformations from a FASTA sequence
and an A3M multiple-sequence alignment (MSA). It can obtain an initial structure
with the bundled OpenFold inference code or start from a user-provided PDB.

The prediction pipeline contains three stages:

1. **Initialization and representation** — OpenFold predicts the initial
   structure, while the selected Xray and/or NMR network processes the MSA.
2. **Geometric exploration** — the predicted geometry is iteratively combined
   with structural distances to obtain updated starting conformations.
3. **Flow sampling** — FlowFormer combines structural, pair, and MSA
   representations before the structure module generates the final ensemble.

Geometric exploration can be disabled when only the initial-structure sampling
paths are required.

## Installation

Clone or download this repository, then enter its root directory:

```bash
cd trFlow
```

It is recommended to use `mamba` to manage the Python dependencies. Follow the
[Mamba installation documentation](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to install it. [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install)
can also be used, but Mamba is usually much faster. If Conda is already
installed, install Mamba into the base environment to avoid conflicts:

```bash
conda install -n base mamba -c conda-forge
```

Create the trFlow environment and install the command-line entry point:

```bash
mamba env create -f environment.yml
mamba activate trflow
python -m pip install -e .
cp config/env_config.json.template config/env_config.json
```

The environment uses the PyTorch 2.6.0 CUDA 11.8 wheel and requires a Linux
machine with an NVIDIA GPU and a compatible driver. `environment.yml` is the
canonical dependency definition. For a pip-only setup, create a Python 3.11
environment, run `python -m pip install -r requirements.txt`, then install
the command with `python -m pip install -e . --no-deps`.

### Model files

Place the following model files in `models/`:

| File | Description |
| --- | --- |
| `trflow_xray.pth` | trFlow Xray checkpoint |
| `trflow_nmr.pth` | trFlow NMR checkpoint |
| `esm_msa1_t12_100M_UR50S.pt` | ESM-MSA-1b weights |
| `openfold_params_model_5_ptm.npz` | OpenFold `model_5_ptm` parameters |

The default configuration expects the files above in `models/`. The local
`config/env_config.json` is intentionally ignored by Git; create it from the
tracked template as shown above, then edit it when using different paths or
GPU indices.

## Usage

The command-line interface follows the form:

```text
trFlow predict INPUT [OPTIONS]
```

`INPUT` may be either a JSON run configuration or a FASTA file.

### Predict from JSON

Run the standard single-target example (`8CRJ_8CRI`) or the two-target
batch example (`8CRJ_8CRI` and `6HKR_7OXW`). Both configurations run the
complete pipeline and generate 10 conformations per target:

```bash
trFlow predict example/example_input.json
trFlow predict example/example_batch_input.json
```

### Predict directly from FASTA and A3M

```bash
trFlow predict example/fasta/6HKR_7OXW.fasta \
  --msa example/msa/6HKR_7OXW.a3m --output-dir outputs \
  --models Xray,NMR --sample-num 10
```

To start from an existing structure instead of running OpenFold:

```bash
trFlow predict sequence.fasta --msa alignment.a3m --init-pdb initial.pdb --output-dir outputs
```

Run `trFlow predict --help` for all available options.

The source-tree interfaces remain available and accept the same prediction
arguments:

```bash
python -m trFlow predict example/example_input.json
python run_trflow.py predict example/example_input.json
```

### Common options

| Option | Description |
| --- | --- |
| `--msa FILE` | A3M alignment for FASTA input |
| `--name NAME` | Sample name for FASTA input |
| `--init-pdb FILE` | Use an existing initial structure |
| `--env FILE` | Use a different environment configuration |
| `--output-dir DIR` | Override the output directory |
| `--sample-num N` | Number of conformations to generate |
| `--models Xray,NMR` | Select one or both trained models |
| `--single-step` | Use one structure-model forward per sample |
| `--steps N` | Set the flow schedule length |
| `--no-random-step` | Use `--steps` instead of random step selection |
| `--no-random-step-size` | Use evenly spaced flow steps |
| `--no-geometric-exploration` | Disable geometric exploration |
| `--parallel` | Enable intra-sample multi-GPU parallelism |
| `--save-repr-npz` | Save intermediate representations |
| `--seed N` | Set an optional reproducibility seed |

No seed is fixed by default. Specify `--seed` or a JSON `seed` value only when
a reproducible run is required. The effective run seed is recorded in
`info.json`.

## JSON input

```json
{
  "output_dir": "outputs",
  "samples": [
    {
      "name": "8CRJ_8CRI",
      "fasta_path": "example/fasta/8CRJ_8CRI.fasta",
      "msa_path": "example/msa/8CRJ_8CRI.a3m",
      "init_pdb": null
    }
  ],
  "options": {
    "sample_num": 10,
    "geometric_exploration": true,
    "single_step": false,
    "steps": 2,
    "random_step": true,
    "random_step_size": true,
    "models": ["Xray", "NMR"],
    "parallel": false,
    "save_repr_npz": false
  }
}
```

Multiple entries in `samples` are processed as a batch. Values supplied on the
command line override the corresponding JSON options.

## Output

Each target is written to `{output_dir}/{sample_name}/`:

```text
{output_dir}/{sample_name}/
|-- input.json
|-- info.json
|-- {sample_name}_init.pdb
|-- {sample_name}_updated_Xray.pdb
|-- {sample_name}_updated_NMR.pdb
|-- {sample_name}_reprs.npz
|-- openfold_log.txt
`-- predictions/
    `-- {sample_name}_sample_001.pdb ...
```

The repository's `outputs/` directory contains the complete 10-structure
results for both included examples. Machine-specific `openfold_log.txt` files
are not committed with these curated outputs. Files that do not apply to the
selected options are omitted. `input.json`
stores the resolved run configuration, while `info.json` records stage timing,
initialization source, generated structures, model paths, and mean pLDDT values.

## Evaluation

The integrated `evaluate` command compares predicted structures with one or
more reference PDB files and writes RMSD and TM-score results to CSV:

```bash
trFlow evaluate --pred-dir outputs/8CRJ_8CRI/predictions --native-dir path/to/references --output outputs/evaluation.csv
```

TM-score calculations always use sequence alignment (`-seq`). By default the
command uses the bundled Linux x86-64 executable at `bin/TMscore`; use
`--tmscore /path/to/TMscore` to explicitly select another executable (for
example, on another platform). The original standalone interface is also
retained:

```bash
python evaluate.py --pred-dir outputs/8CRJ_8CRI/predictions --native-dir path/to/references
```

## Tests

Run the lightweight unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests do not run GPU inference; they cover the TMscore command behavior
and portable output-path serialization. The complete GPU examples are the JSON
configurations described above.

## Acknowledgements

This repository vendors portions of OpenFold and ESM used by the inference
pipeline. Their original copyright and license notices are retained in the
corresponding source files. Model weights are distributed separately.

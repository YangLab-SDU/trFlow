<h1 align="center">trFlow: ultrafast all-atom protein conformational ensemble generation via conditional flow matching</h1>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch 2.6.0" src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="LICENSE"><img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache%202.0-D22128?style=flat-square"></a>
</p>

<p align="center">
  <img src="assets/trflow_workflowv2.png" alt="Overview of the trFlow conformational ensemble generation workflow" width="100%">
</p>

## Overview

trFlow generates alternative protein conformations from an A3M
multiple-sequence alignment (MSA). The first A3M record supplies the target
sequence; an optional FASTA file can be provided for strict cross-validation.
trFlow can obtain an initial structure with the bundled OpenFold inference code
or start from a user-provided PDB.

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

### Step 1. Clone the repository

```bash
git clone https://github.com/YangLab-SDU/trFlow.git
cd trFlow
```

### Step 2. Download the model weights

Download the pretrained models and extract the four files directly into
`models/`:

```bash
mkdir -p models
wget http://yanglab.qd.sdu.edu.cn/trFlow/pretrained_models.tar.bz2
tar -xjf pretrained_models.tar.bz2 -C models
```

The archive contains:

| File | Description |
| --- | --- |
| `trflow_xray.pth` | trFlow Xray checkpoint |
| `trflow_nmr.pth` | trFlow NMR checkpoint |
| `esm_msa1_t12_100M_UR50S.pt` | ESM-MSA-1b weights |
| `openfold_params_model_5_ptm.npz` | OpenFold `model_5_ptm` parameters |

The default configuration expects these files in `models/`.

### Step 3. Install the environment

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
machine with an NVIDIA GPU and a compatible driver. `environment.yml` defines
the tested environment. For a pip-only setup, create a Python 3.11 environment,
run `python -m pip install -r requirements.txt`, then install the command with
`python -m pip install -e . --no-deps`.

To use different model paths or GPU indices, edit the local
`config/env_config.json` created above.

## Usage

The command-line interface follows the form:

```text
trFlow predict INPUT [OPTIONS]
```

`INPUT` may be a JSON run configuration or an A3M file. A FASTA file can also
be used together with `--msa`; direct A3M input is recommended.

### Predict from JSON

Run the standard single-target example (`8CRJ_8CRI`) or the two-target
batch example (`8CRJ_8CRI` and `2akl`). Both configurations run the
complete pipeline and generate 10 conformations per target:

```bash
trFlow predict example/example_input.json
trFlow predict example/example_batch_input.json
```

### Predict directly from A3M

```bash
trFlow predict example/msa/2akl.a3m \
  --output-dir outputs \
  --models Xray,NMR --sample-num 10
```

The first A3M record must be the ungapped target sequence. To cross-check it
against a separate FASTA file:

```bash
trFlow predict example/msa/8CRJ_8CRI.a3m \
  --fasta example/fasta/8CRJ_8CRI.fasta --output-dir outputs
```

To start from an existing structure instead of running OpenFold:

```bash
trFlow predict alignment.a3m --init-pdb initial.pdb --output-dir outputs
```

To provide the target sequence explicitly, use a FASTA file together with its
A3M alignment:

```bash
trFlow predict sequence.fasta --msa alignment.a3m --output-dir outputs
```

Run `trFlow predict --help` for all available options.
By default, trFlow generates 200 conformations per target; use `--sample-num`
to select a different ensemble size.

### Common options

| Option | Description |
| --- | --- |
| `--fasta FILE` | Optional FASTA used to validate the first A3M record |
| `--msa FILE` | A3M alignment when a FASTA file is used as `INPUT` |
| `--name NAME` | Set the target name for direct A3M/FASTA input |
| `--init-pdb FILE` | Use an existing initial structure |
| `--env FILE` | Use a custom environment configuration |
| `--output-dir DIR` | Set the output directory |
| `--sample-num N` | Number of conformations to generate (default: 200) |
| `--models Xray,NMR` | Select one or both trained models |
| `--single-step` | Use one structure-model forward per sample |
| `--steps N` | Set the number of structure-model forwards |
| `--no-random-step` | Use `--steps` instead of randomly choosing 1 or 7 forwards |
| `--no-random-step-size` | Use evenly spaced flow times |
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
      "msa_path": "example/msa/8CRJ_8CRI.a3m",
      "init_pdb": null
    }
  ],
  "options": {
    "sample_num": 10,
    "geometric_exploration": true,
    "single_step": false,
    "steps": 7,
    "random_step": true,
    "random_step_size": true,
    "models": ["Xray", "NMR"],
    "parallel": false,
    "save_repr_npz": false
  }
}
```

Multiple entries in `samples` are processed as a batch. Command-line options
take precedence over the corresponding JSON values.

The optional `fasta_path` field enables strict sequence cross-validation.
When it is omitted, the target sequence is read from the first A3M record.
FASTA/A3M sequence mismatches are rejected.

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

Ready-to-inspect 10-structure results for both examples are included in
`outputs/`. Files that do not apply to the selected options are omitted.
`input.json` stores the resolved run configuration, while `info.json` records
stage timing, initialization source, generated structures, model paths, and
mean pLDDT values.

## Evaluation

`trFlow evaluate` compares predicted structures with one or more reference PDB
files and writes RMSD and TM-score results to CSV:

```bash
trFlow evaluate --pred-dir outputs/8CRJ_8CRI/predictions --native-dir path/to/references --output outputs/evaluation.csv
```

TM-score calculations always use sequence alignment (`-seq`). By default the
command uses the bundled Linux x86-64 executable at `bin/TMscore`. Use
`--tmscore /path/to/TMscore` to select another executable, for example on a
different platform.

## Tests

Run the lightweight unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

These tests do not run GPU inference; they cover direct A3M input, sequence
validation, TMscore command behavior, and portable output-path serialization.
The complete GPU examples are the JSON configurations described above.

## Acknowledgements

This repository vendors portions of OpenFold and ESM used by the inference
pipeline. Their original copyright and license notices are retained in the
corresponding source files. Model weights are distributed separately.

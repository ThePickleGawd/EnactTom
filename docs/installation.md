# Installation

This page gives the complete setup for EnactToM. The top-level README contains
the short local setup; this file adds the Habitat simulator and asset steps
needed for scene generation, replay, and benchmarking.

## Local Setup

The local setup runs task validation, PDDL solving, task-generation utilities,
the default `mini` authoring agent, and the test suite:

```bash
conda create -n enacttom python=3.10 cmake=3.14.0 -y
conda activate enacttom
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use `mamba` instead of `conda` if it is available.

Check the install:

```bash
bash -n enacttom/run.sh
python -m compileall -q enacttom habitat_llm tests
python -m pytest
./enacttom/run.sh --help
```

## Habitat Setup

Habitat execution requires Linux, GPU/CUDA drivers for CUDA runs, `git-lfs`,
and either `conda` or `mamba`.

Initialize Git LFS once on the machine:

```bash
git lfs install
```

Install PyTorch and Habitat-Sim in the active `enacttom` environment. Adjust
the CUDA package if the target machine uses a different CUDA runtime:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
```

Install Habitat-Lab/Baselines from the revision used for this release, then
reinstall EnactToM:

```bash
HABITAT_LAB_COMMIT=094d6be2f9d057e4781a68ae792132895fd4d3d0

python -m pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-lab"
python -m pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-baselines"
python -m pip install -r requirements.txt
python -m pip install -e .
```

If dynamic libraries fail to load, make sure the active conda environment's
`lib` directory is on `LD_LIBRARY_PATH`.

## Assets

Download Habitat robot and rearrangement assets:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids rearrange_task_assets hab_spot_arm hab3-episodes \
  --data-path data/ \
  --no-replace \
  --no-prune
```

Download the object asset library:

```bash
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
git -C data/objects_ovmm lfs pull
```

Download HSSD. The upstream branch is still named `partnr`; the local paths are
the EnactToM paths expected by the retained configs:

```bash
mkdir -p data/versioned_data
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
git -C data/versioned_data/hssd-hab lfs pull
ln -sfn versioned_data/hssd-hab data/hssd-hab
```

Provide the EnactToM episode file expected by
`habitat_llm/conf/habitat_conf/dataset/enacttom_hssd.yaml`:

```bash
mkdir -p data/datasets/enacttom_episodes/v0_0
# Place train_2k.json.gz here:
# data/datasets/enacttom_episodes/v0_0/train_2k.json.gz
```

Alternatively, point the runner at another compatible episode file:

```bash
export ENACTTOM_EPISODES_PATH=/absolute/path/to/train_2k.json.gz
```

Check required assets:

```bash
test -f data/hssd-hab/hssd-hab-enacttom.scene_dataset_config.json
test -f "${ENACTTOM_EPISODES_PATH:-data/datasets/enacttom_episodes/v0_0/train_2k.json.gz}"
test -d data/objects_ovmm/train_val/hssd/configs/objects
test -f data/robot_variants/hab_spot_arm/urdf/hab_spot_arm_agent_0_scarlet.urdf
```

## Model APIs

Configure credentials through environment variables or a repo-root `.env` file:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

AWS Bedrock users should also install the optional client:

```bash
python -m pip install boto3
```

## Habitat Checks

Run Habitat-backed checks only after the asset checks pass:

```bash
./enacttom/run.sh new-scene --agents 2 --output-dir /tmp/enacttom-scene
./enacttom/run.sh generate --num-tasks 1 --difficulty standard
./enacttom/run.sh benchmark --tasks-dir data/enacttom/tasks --model gpt-5.4 --num-times 3
```

Missing Habitat dependencies or assets are setup errors and should be fixed
before running `generate`.

Legacy task datasets, neural-network skill checkpoints, semantic map/RAG data,
humanoid assets, and submodules are not required for the trimmed EnactToM paper
path.

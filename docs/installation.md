# Installation

This is the full setup path for running EnactToM with Habitat. Use `uv` for Python package installation, but keep `conda` or `mamba` for the Habitat-Sim and PyTorch binary stack.

## Prerequisites

- `git-lfs`
- `uv`
- `conda` or `mamba`
- GPU/CUDA drivers if running the benchmark on Linux with CUDA

## Environment

Create the Habitat runtime environment:

```bash
mamba create -n enacttom python=3.9.2 cmake=3.14.0 -y
mamba activate enacttom
```

Install PyTorch and Habitat-Sim. Adjust the CUDA package to match the target machine:

```bash
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
mamba install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
```

Install Habitat-Lab/Baselines from the upstream revisions this repo was cleaned against, then install EnactToM:

```bash
HABITAT_LAB_COMMIT=094d6be2f9d057e4781a68ae792132895fd4d3d0

uv pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-lab"
uv pip install "git+https://github.com/facebookresearch/habitat-lab.git@${HABITAT_LAB_COMMIT}#subdirectory=habitat-baselines"
uv pip install -r requirements.txt
uv pip install -e .
```

If using AWS Bedrock instead of OpenAI/Anthropic direct APIs, install the optional AWS client:

```bash
uv pip install boto3
```

If dynamic libraries fail to load, make sure the active conda environment library directory is on `LD_LIBRARY_PATH`.

## Assets

Download Habitat robot/rearrangement assets:

```bash
python -m habitat_sim.utils.datasets_download \
  --uids rearrange_task_assets hab_spot_arm hab3-episodes \
  --data-path data/ \
  --no-replace \
  --no-prune
```

Download the object asset library used by the retained configs:

```bash
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
git -C data/objects_ovmm lfs pull
```

Download HSSD. The upstream branch is still named `partnr`; the local paths and code path are EnactToM:

```bash
mkdir -p data/versioned_data
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
git -C data/versioned_data/hssd-hab lfs pull
ln -sfn versioned_data/hssd-hab data/hssd-hab
```

Provide the EnactToM episode dataset expected by `habitat_llm/conf/habitat_conf/dataset/enacttom_hssd.yaml`:

```bash
mkdir -p data/datasets/enacttom_episodes/v0_0
# Place train_2k.json.gz here:
# data/datasets/enacttom_episodes/v0_0/train_2k.json.gz
```

Alternatively, point the runner at another compatible episode file:

```bash
export ENACTTOM_EPISODES_PATH=/absolute/path/to/train_2k.json.gz
```

Check the required assets:

```bash
test -f data/hssd-hab/hssd-hab-enacttom.scene_dataset_config.json
test -f "${ENACTTOM_EPISODES_PATH:-data/datasets/enacttom_episodes/v0_0/train_2k.json.gz}"
test -d data/objects_ovmm/train_val/hssd/configs/objects
test -f data/robot_variants/hab_spot_arm/urdf/hab_spot_arm_agent_0_scarlet.urdf
```

## Credentials

Configure model credentials through environment variables or a repo-root `.env` file:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

## Smoke Checks

Run lightweight checks:

```bash
bash -n enacttom/run_enacttom.sh
python -m compileall -q enacttom habitat_llm tests
./enacttom/run_enacttom.sh --help
```

Run the Habitat-backed flow only after the asset checks above pass:

```bash
./enacttom/run_enacttom.sh new-scene --agents 2 --output-dir /tmp/enacttom-scene
./enacttom/run_enacttom.sh generate --num-tasks 1 --difficulty standard
./enacttom/run_enacttom.sh benchmark --tasks-dir data/enacttom/tasks --model gpt-5.4 --num-times 3
```

Legacy task datasets, neural-network skill checkpoints, semantic map/RAG data, humanoid assets, and submodules are not required for the trimmed EnactToM paper path.

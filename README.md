# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

## Overview

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing NeurINO, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

## Installation
### 1. Clone this repository

```bash
git clone https://github.com/your-username/neurino.git
cd neurino```

### 2. Create a conda environment
conda create -n neurino python=3.10 -y
conda activate neurino

3. Install PyTorch

Please follow the official PyTorch instructions depending on your OS and CUDA version:

# Example only — check the official PyTorch website for the correct command
pip install torch torchvision torchaudio

4. Install additional dependencies
pip install -r requirements.txt


Or for editable installation:

pip install -e .

Data

We evaluate NeurINO on the following datasets:

BigNeuron – Drosophila

BigNeuron – Mouse

NeuroFly

CWMBS

⚠️ Raw datasets are not included in this repository.
Please download the data from their official sources.

Recommended directory structure
data/
├── BigNeuron/
│   ├── Drosophila/
│   │   ├── images/
│   │   └── swc/
│   └── Mouse/
│       ├── images/
│       └── swc/
├── NeuroFly/
│   ├── images/
│   └── swc/
└── CWMBS/
    ├── images/
    └── swc/


If you provide train/val/test splits:

splits/
├── bigneuron_drosophila_train.txt
├── bigneuron_drosophila_val.txt
└── bigneuron_drosophila_test.txt

Configuration files

Dataset settings (root path, patch size, augmentations…) can be defined under:

configs/
├── neurino_drosophila.yaml
├── neurino_mouse.yaml
├── neurino_neurofly.yaml
└── neurino_cwmbs.yaml

Usage

All commands below assume you are in the project root directory.

cd /path/to/neurino


The following examples serve as draft templates — you can adjust paths and arguments once your actual code structure is finalized.

Preprocessing

Preprocessing typically includes normalization, resampling, cropping, patch extraction, and optional skeleton computation.

python scripts/preprocess.py \
    --config configs/neurino_drosophila.yaml \
    --dataset bigneuron_drosophila \
    --split_dir splits/ \
    --output_dir data/processed/BigNeuron/Drosophila


Common arguments:

--config: YAML file describing dataset & model settings

--dataset: dataset name

--split_dir: directory of split text files

--output_dir: where processed data will be stored

Training

We consider two variants of NeurINO:

NeurINO-T — Tiny DINOv3 backbone

NeurINO-S — Small DINOv3 backbone

Example training command:

python scripts/train.py \
    --config configs/neurino_drosophila.yaml \
    --backbone dino_convnext_tiny \
    --exp_name neurino_tiny_drosophila \
    --gpus 0


Or multi-GPU training:

python scripts/train.py \
    --config configs/neurino_neurofly.yaml \
    --backbone dino_convnext_small \
    --exp_name neurino_small_neurofly \
    --gpus 0,1


Common arguments:

--config: path to YAML config

--backbone: backbone type

--exp_name: experiment name

--gpus: GPU IDs

--resume: resume from checkpoint

--num_epochs: override number of epochs

--amp: enable mixed precision

Evaluating
1. Segmentation metrics
python scripts/evaluate.py \
    --config configs/neurino_drosophila.yaml \
    --checkpoint checkpoints/neurino_tiny_drosophila/best.ckpt \
    --eval_mode seg \
    --save_dir outputs/drosophila_seg


Outputs may include:

F1-score

HD95

Per-volume metrics

Aggregated dataset metrics

2. Reconstruction metrics (using external tracers)
python scripts/evaluate.py \
    --config configs/neurino_drosophila.yaml \
    --checkpoint checkpoints/neurino_tiny_drosophila/best.ckpt \
    --eval_mode trace \
    --tracer smarttracing \
    --tracer_exe /path/to/SmartTracing \
    --save_dir outputs/drosophila_trace

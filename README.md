# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

## Overview

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing NeurINO, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

## Installation
### 1. Clone this repository

```bash
git clone https://github.com/your-username/neurino.git
cd neurino
```

### 2. Create a conda environment

```bash
conda create -n neurino python=3.10 -y
conda activate neurino
```

### 3. Install PyTorch

```bash
# Example — please check https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Or editable installation:

```bash
pip install -e .
```

## Data

This project supports four 3D neuron datasets:

- **BigNeuron – Drosophila**
- **BigNeuron – Mouse**
- **NeuroFly**
- **CWMBS**

Raw datasets are **not included**; please download from official sources.

### Recommended directory structure

```text
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
```

### Split files (optional)

```text
splits/
├── bigneuron_drosophila_train.txt
├── bigneuron_drosophila_val.txt
└── bigneuron_drosophila_test.txt
```

### Configuration files

```text
configs/
├── neurino_drosophila.yaml
├── neurino_mouse.yaml
├── neurino_neurofly.yaml
└── neurino_cwmbs.yaml
```

## Usage

All commands assume you are in the project root directory:

```bash
cd /path/to/neurino
```

Commands below are **draft templates**. We will update them once your actual code structure is finalized.

### Preprocessing

```bash
python scripts/preprocess.py \
    --config configs/neurino_drosophila.yaml \
    --dataset bigneuron_drosophila \
    --split_dir splits/ \
    --output_dir data/processed/BigNeuron/Drosophila
```

### Training

```bash
python scripts/train.py \
    --config configs/neurino_drosophila.yaml \
    --backbone dino_convnext_tiny \
    --exp_name neurino_tiny_drosophila \
    --gpus 0
```

Multi-GPU:

```bash
python scripts/train.py \
    --config configs/neurino_neurofly.yaml \
    --backbone dino_convnext_small \
    --exp_name neurino_small_neurofly \
    --gpus 0,1
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/neurino_drosophila.yaml \
    --checkpoint checkpoints/neurino_tiny_drosophila/best.ckpt \
    --eval_mode seg \
    --save_dir outputs/drosophila_seg
```

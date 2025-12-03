# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

## Overview

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing **NeurINO**, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

## Installation
### 1. Clone this repository

```bash
git clone https://github.com/yy0007/NeurINO.git
cd NeurINO
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

The datasets used in our paper are from BigNeuron, NeuroFly and CWMBS. 

## Usage

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

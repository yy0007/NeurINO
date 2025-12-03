# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

## Overview

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing NeurINO, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

## Installation
### 1. Clone this repository

```bash
git clone https://github.com/your-username/neurino.git
cd neurino

2. Create a conda environment
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

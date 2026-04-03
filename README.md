# 🧠 NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization 

<p align="center">
  <a href="https://arxiv.org/abs/XXXX"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a>
  <a href="https://github.com/yy0007/NeurINO"><img src="https://img.shields.io/badge/Code-GitHub-black"></a>
  <a href="https://yy0007.github.io/NeurINO"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
</p>

# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing **NeurINO**, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

NeurINO is developed on top of [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) framework. We evaluate NeurINO on three neuronal imaging datasets:

- [BigNeuron](https://github.com/BigNeuron/Data/releases)
- [NeuroFly](https://zenodo.org/records/13328867) 
- [CWMBS](https://github.com/crz22/CWMBS)  

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

### 3. Install dependencies

```bash
pip install -e .
```

## Usage

Our framework follows the MedNeXt-style preprocessing and training workflows. 

### Preprocessing

Run data preprocessing with:

```bash
neurino_plan_and_preprocess \
    -t 001 \
    -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
```

Where:

- `-t 001` = task ID 
- `-pl3d` = 3D planner variant  

### Training

General training command: 

```bash
neurino_train \
    3d_fullres <TrainerName> <TASK_ID> <FOLD> \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

#### Trainer naming convention

| Component | Meaning |
|----------|---------|
| `CenterInfla` / `AvgInfla` | Inflation strategy (center vs. average) | 
| `T` / `S` | DINOv3-Tiny / DINOv3-Small backbone |

#### Examples

##### Center inflation + DINOv3-Tiny

```bash
neurino_train \
    3d_fullres NeurINO_CenterInfla_SGL_T_kernel3 001 0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

##### Average inflation + DINOv3-Small

```bash
neurino_train \
    3d_fullres NeurINO_AvgInfla_SGL_S_kernel3 001 0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

### Inference

Inference / evaluation is done with:

```bash
neurino_predict \
    -i <imagesTs_dir> \
    -o <output_dir> \
    -t <TASK_ID> \
    -m 3d_fullres \
    -f <FOLD> \
    -p nnUNetPlansv2.1_trgSp_1x1x1 \
    -tr <TrainerName> \
    -chk model_best
``` 

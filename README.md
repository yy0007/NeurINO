# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

## Overview

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing **NeurINO**, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D neuron domain.

NeurINO is developed on top of MedNeXt. We evaluate NeurINO on three neuronal imaging datasets:

- [BigNeuron](#bigneuron)
- [NeuroFly](#neurofly)
- [CWMBS](#cwmbs) 

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

### Preprocessing

Preprocessing follows the MedNeXt planning & preprocessing pipeline:

```bash
mednextv1_plan_and_preprocess \
    -t 001 \
    -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1
```

Where:

- `-t 001` = task ID 
- `-pl3d` = 3D planner variant (target spacing 1×1×1) 

### Training

Training uses the nnUNet_mednext trainer system.

General pattern:

```bash
python -m nnunet_mednext.run.run_training \
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
python -m nnunet_mednext.run.run_training \
    3d_fullres NeurINO_CenterInfla_SGL_T_kernel3 001 0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

##### Average inflation + DINOv3-Tiny

```bash
python -m nnunet_mednext.run.run_training \
    3d_fullres NeurINO_AvgInfla_SGL_T_kernel3 001 0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

##### Center inflation + DINOv3-Small

```bash
python -m nnunet_mednext.run.run_training \
    3d_fullres NeurINO_CenterInfla_SGL_S_kernel3 001 0 \
    -p nnUNetPlansv2.1_trgSp_1x1x1
```

### Evaluation

Inference / evaluation is done with:

```bash
python -m nnunet_mednext.inference.predict_simple \
    -i <imagesTs_dir> \
    -o <output_dir> \
    -t <TASK_ID> \
    -m 3d_fullres \
    -f <FOLD> \
    -p nnUNetPlansv2.1_trgSp_1x1x1 \
    -tr <TrainerName> \
    -chk model_best
``` 

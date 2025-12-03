## NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

1. Overview
High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing NeurINO, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model DINOv3 into the 3D domain.

# NeurINO: NeuroSeg Meets DINOv3

## 🧠 NeurINO: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOV3 Initialization

**CVPR 2026 Submission (Paper ID: 8693)**

[TODO: Place a representative image, such as the model architecture figure or a side-by-side visualization of segmentation results, here.]

---

## 1. Overview

This repository contains the official PyTorch implementation for our CVPR 2026 submission, **"NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOV3 Initialization."**

High-quality 3D neuron segmentation is critical for neuroscience, but its progress is hampered by the scarcity of annotated 3D volumetric data. We address this by proposing **NeurINO**, a novel framework that effectively transfers the rich 2D visual representations learned by the self-supervised foundation model **DINOv3** into the 3D domain.

### Key Contributions:

* **DINOv3 Prior Adaptation:** We are the first to successfully leverage the powerful DINOv3 visual prior for 3D neuroimaging segmentation.
* **Inflation-Based 2D-3D Adaptation:** We introduce a straightforward yet effective inflation strategy to adapt 2D DINOv3 weights into 3D convolutional kernels, enabling efficient transfer learning while focusing on 3D spatial continuity.
* **Topology-Aware Skeleton Loss (TASL):** To enforce morphological fidelity, especially for complex and slender neuronal structures, we propose TASL, which explicitly penalizes structural discrepancies at the **node, edge, and path levels** of the skeletonized segmentations.
* **State-of-the-Art (SOTA) Results:** NeurINO achieves superior performance across challenging datasets (e.g., BigNeuron, NeuroFly, CWMBS), demonstrating significant gains in topology-aware metrics like ESA, DSA, and PDS.

---

## 2. Installation

### Prerequisites

* Linux / Windows
* Python 3.9+
* PyTorch (Tested with 2.0.1 and CUDA 11.8)

### Setup

1.  **Clone the Repository:**
    ```bash
    git clone [TODO: Your GitHub Repository Link]
    cd NeurINO
    ```

2.  **Create and Activate Conda Environment (Recommended):**
    ```bash
    conda create -n neurino python=3.9
    conda activate neurino
    ```

3.  **Install PyTorch:**
    ```bash
    # Please check PyTorch official site for your specific CUDA version
    pip install torch torchvision torchaudio
    ```

4.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    # [TODO: Ensure requirements.txt includes: numpy, h5py, scikit-image, opencv-python, tqdm, einops, etc., including any necessary 3D skeletonization library.]
    ```

---

## 3. Data

We evaluated NeurINO on four publicly available 3D neuron datasets: **BigNeuron (Drosophila & Mouse), NeuroFly, and CWMBS**.

### Data Preparation Steps:

1.  **Download Datasets:** Please download the raw volumetric image data and corresponding ground-truth segmentation masks from their respective official sources.
2.  **Organize Data Structure:** Place the processed data under the `./data` directory with the following structure:
    ```
    ./data/
    ├── BigNeuron_Drosophila/
    │   ├── volumes/
    │   └── masks/
    ├── BigNeuron_Mouse/
    │   ├── volumes/
    │   └── masks/
    ├── NeuroFly/
    │   ...
    └── CWMBS/
        ...
    # [TODO: Specify your exact required file naming conventions (e.g., .h5 or .nii.gz)]
    ```
3.  **Initial Preprocessing:** Ensure your data is normalized (e.g., to $[0, 1]$ or standardized) and handled according to your paper's description before the `Preprocessing` step below.

---

## 4. Usage

### 4.1. Preprocessing

Before training, we perform several data augmentation and patch-sampling steps. The following script can be used to generate the necessary training/validation/testing splits and process the volumes into the format expected by the dataloader.

```bash
# Example: Generate patches and lists for the Drosophila dataset
python data_utils/prepare_data.py --dataset Drosophila --output_dir ./data_processed
# [TODO: Replace with your actual data preparation script and parameters]

# 🤖 NeurINO AutoResearch

<p align="center">
  Autonomous hypothesis-driven research framework for topology-aware 3D neuron segmentation.
</p>

<!-- <p align="center">
  <a href="../README.md"><img src="https://img.shields.io/badge/Main-NeurINO-black"></a>
  <a href="./program_seg.md"><img src="https://img.shields.io/badge/Research-Policy-blue"></a>
</p> -->

---

## 🧠 Overview

NeurINO AutoResearch is an autonomous experimentation framework designed for controlled and interpretable exploration of:

- DINOv3-to-3D transfer strategies
- topology-aware learning objectives
- architectural configurations
- training and optimization strategies

This framework operates within a constrained interpretable research space designed for:
- scientific experimentation,
- topology-aware reasoning,
- and interpretable optimization.

---

## ⚙️ Workflow

The AutoResearch pipeline:

1. modifies a restricted set of experiment parameters,
2. launches NeurINO training,
3. evaluates validation performance,
4. records experiment metrics,
5. iteratively explores improved configurations.

The framework is intentionally conservative:
- core code is not modified,
- dataset preprocessing and evaluation remain fixed,
- only predefined research variables are explored.

---

## 📂 Main Components

| File | Description |
|---|---|
| `train_seg_autoresearch.py` | Autonomous experiment launcher |
| `program_seg.md` | Research policy and experiment strategy |
| `autoresearch_runs/` | Experiment outputs and compact metrics |

---

## 🔬 Search Space

### 🧬 DINO / Transfer Learning
- inflation strategies
- DINOv3 backbone variants
- kernel inflation depth
- encoder freezing

### 🏗️ Architecture
- model block configuration
- kernel sizes
- deep supervision

### 🌱 Topology-aware Learning
- topology-aware losses
- loss weighting
- topology-sensitive optimization

---

## 🚀 Example Usage

Run from the project root:

```bash
python autoresearch/train_seg_autoresearch.py
```

Experiment summaries are written to:

```text
autoresearch_runs/ 
```

---

## 📊 Experiment Philosophy

This framework is designed for:
- hypothesis-driven experimentation,
- topology-aware optimization,
- controlled scientific exploration,
- and interpretable analysis.

The goal is not brute-force hyperparameter search, but interpretable and research-oriented experimentation.

---

## 📌 Notes

- Only predefined research variables are explored.
- Core code remains fixed.
- The framework is intended for research experimentation rather than large-scale black-box hyperparameter optimization. 

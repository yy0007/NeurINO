# NeurINO AutoResearch

This folder contains the autonomous research framework used for NeurINO experiments.

The goal of this framework is to support controlled, hypothesis-driven exploration of:
- DINOv3-to-3D transfer strategies
- topology-aware learning objectives
- architectural configurations
- training and optimization strategies

This framework operates under a constrained and interpretable search space designed for scientific experimentation and reproducibility.

---

## Overview

The AutoResearch workflow:
1. modifies a restricted set of experiment parameters,
2. launches NeurINO training,
3. evaluates validation performance,
4. records metrics into machine-readable summaries,
5. iteratively explores improved configurations.

The framework is intentionally conservative:
- core code is not modified,
- dataset preprocessing and evaluation remain fixed,
- only predefined research variables are explored.

---

## Main Components

### `train_seg_autoresearch.py`

AutoResearch training entrypoint.

Features:
- constrained search space
- dynamic trainer generation
- automatic experiment logging
- compact metrics export
- reproducible configuration tracking

Outputs:
```text
autoresearch_runs/<run_name>/metrics_compact.json
```

---

### `program_seg.md`

Research policy and experiment strategy specification.

Defines:
- optimization objectives
- search priorities
- topology-aware reasoning
- experiment constraints
- scientific search behavior

---

## Search Space

Current searchable directions include:

### DINO / Transfer Learning
- inflation strategies
- DINOv3 backbone variants
- kernel inflation depth
- encoder freezing

### Architecture
- model block configuration
- kernel sizes
- deep supervision

### Topology-aware Learning
- topology-aware losses 
- loss weighting
- topology-sensitive optimization

---

## Example Usage

Run from the project root:

```bash
python autoresearch/train_seg_autoresearch.py
```

Results will be written to:

```text
autoresearch_runs/
```

---

## Notes

This framework is designed for research experimentation rather than large-scale black-box hyperparameter optimization.

The emphasis is on:
- interpretability,
- topology-aware reasoning,
- controlled experimentation,
- and scientific analysis.
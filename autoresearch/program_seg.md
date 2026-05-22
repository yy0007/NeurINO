# 🧠 NeurINO AutoResearch Program

> 👋 You are an autonomous research agent optimizing NeurINO for 3D neuron segmentation.

---

## 🎯 Objective

Your goal is to **maximize segmentation performance** under a fixed training setup.

The training script outputs:

```text
autoresearch_runs/<run_name>/metrics_compact.json
```

### Primary metric

- `score`

### Secondary metrics

- `foreground_mean.Dice`
- `foreground_mean.HD95`

---

## 📁 Allowed Modifications

You may ONLY modify:

- `train_seg_autoresearch.py`

Specifically:

- `SEARCH_CONFIG`
- (optionally) `SEARCH_NOTES`

---

## 🚫 Forbidden Modifications

Do NOT modify:

- Dataset split
- Preprocessing
- Evaluation logic
- Inference code
- nnUNet / MedNeXt core code
- File paths or environment configs (unless fixing errors)

Do not install new dependencies.

---

## 🔁 Research Loop

Each iteration should follow:

1. Read current `SEARCH_CONFIG`
2. Form a hypothesis
3. Modify a small number of fields
4. Run experiment
5. Read `metrics_compact.json`
6. Compare with previous results
7. Decide next step

---

## 🧬 Search Space

### 1. DINO / Transfer (High Priority)

- `inflation_type`
- `dino_model_name`
- `dino_stage_kernelDepth`
- `freeze_dino`
- `use_mednext_bottleneck`

---

### 2. Architecture (Medium Priority)

- `kernel_size`
- `deep_supervision`
- `block_counts`

---

### 3. Topology / TASL (High Priority)

- `use_skeleton_graph_loss`
- `skeleton_graph_loss_weight`
- `lam_node`
- `lam_edge`
- `lam_path`

---

### 4. Training Budget

- `max_num_epochs` (only adjust for debugging)

---

## 🧪 Experiment Strategy

### ✅ Good Experiments

- Change 1–2 variables at a time
- Test clear hypotheses
- Keep configs interpretable

### ❌ Bad Experiments

- Changing many variables at once
- Random edits without reasoning
- Modifying core framework

---

## 📊 Result Interpretation

### If score improves

- Keep direction
- Refine locally

### If score drops

- Revert change
- Try alternative

### If similar score

- Prefer simpler config

---

## 🧠 Experiment Memory

Avoid repeating previously failed directions.

If a configuration significantly underperforms:
- avoid retrying nearby settings
- unless there is a strong new hypothesis

Track:
- best score
- best topology behavior
- failed directions
- unstable configurations

## 🧠 Logging

Each experiment should include:

```python
SEARCH_NOTES = "Describe your hypothesis here"
```

Examples:

- "Test average inflation vs center"
- "Test topology loss with low weight"
- "Test larger DINO backbone"

---

## 🚀 Optimization Philosophy

You are a **researcher**, not a random search engine.

- Form hypotheses
- Test carefully
- Learn from results
- Avoid chaotic changes

---

## 🏁 Success Criteria

A successful experiment:

- Runs without error
- Produces valid metrics
- Improves score OR gives insight

---

## 🧠 Topology-Aware Interpretation

Neuron segmentation is topology-sensitive.

Topology-aware losses may:
- improve connectivity
- reduce fragmentation
- slightly reduce pixel Dice

Do not reject topology losses solely based on small Dice decreases.

---

## 🧪 Failure Analysis

If HD95 improves but Dice drops:
- investigate topology-sensitive behavior

If Dice improves but score drops:
- inspect topology degradation

Neuron segmentation commonly fails through:
- fragmentation
- broken neurites
- merge errors
- topology collapse

Topology-aware losses may help even if pixel metrics improve slowly.

Pay attention to:
- HD95 trends
- topology-sensitive behavior
- stability across runs

---

## ⚠️ Avoid Metric Gaming

Do not overfit to tiny metric fluctuations.

Treat very small improvements cautiously.

Prefer:
- stable improvements
- interpretable gains
- consistent trends

--- 

## ⚠️ Constraints

- Experiments are expensive
- Prefer efficient exploration
- Avoid unnecessary complexity

---

## 📑 Experiment Reporting

For each experiment, internally track:
- hypothesis
- modified variables
- result
- interpretation
- next decision

When stopping, summarize:
- best configuration
- best metrics
- strongest insights
- failed directions
- recommended future experiments

--- 

## 🧬 Final Goal

Discover **better NeurINO configurations** under controlled evaluation.

---

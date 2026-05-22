#!/usr/bin/env python3
"""
AutoResearch-friendly training entrypoint for NeurINO / MedNeXt.

What this script changes compared with a plain training launcher:
- exposes a SMALL, EXPLICIT search space for an LLM/agent
- validates that only allowed keys are changed
- generates a dynamic trainer class instead of modifying core project files
- logs every run into an experiment directory with config + summary + score
- writes a compact machine-readable metrics file for outer-loop agents

Recommended workflow
--------------------
1. Keep the original nnunet_mednext codebase stable.
2. Let the agent edit only the SEARCH_CONFIG block below.
3. Run this script.
4. Read autoresearch_runs/.../metrics_compact.json in the outer loop.

Notes
-----
- This is a project-level integration skeleton. You may still need to adapt
  task names, environment variables, or metric extraction to your local setup.
- It is intentionally conservative: the agent should search over meaningful
  research variables, not rewrite the entire nnUNet/MedNeXt codebase.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# =============================================================================
# EDITABLE SEARCH CONFIG
# =============================================================================
# Only edit values for keys that already exist here.
# Do not add new keys unless you also update ALLOWED_KEYS / validation logic.
SEARCH_CONFIG: Dict[str, Any] = {
    # ----- experiment routing -----
    "network": "3d_fullres",
    "task": "Task001_MyDataset",
    "fold": 0,
    "plans_identifier": "nnUNetPlansv2.1_trgSp_1x1x1",
    "validation_only": False,
    "continue_training": False,
    "deterministic": False,
    "mixed_precision": True,
    "sample_by_frequency": False,
    "val_folder": "validation_raw_autoresearch",

    # ----- runtime / budgeting -----
    "max_num_epochs": 20,
    "disable_checkpointing": True,

    # ----- experiment bookkeeping -----
    "experiment_name": "neurino_autoresearch",
    "runs_root": "autoresearch_runs",
    "trainer_class_name": "AutoResearchNeurINO",
    "trainer_module_name": "nnUNetTrainerV2_AutoResearchNeurINO",

    # ----- network / mednext knobs -----
    "n_channels": 32,
    "exp_r": 2,
    "kernel_size": 3,
    "deep_supervision": True,
    "do_res": True,
    "do_res_up_down": True,
    "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],

    # ----- DINO transfer knobs -----
    "use_dino_encoder": True,
    "dino_model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "freeze_dino": True,
    "dino_train_from_scratch": False,
    "dino_stage_kernelDepth": 3,
    "dino_downsample0_to_double": True,
    "inflation_type": "average",  # center / average
    "use_mednext_bottleneck": True,
    "batchrenorm3d_replace_scope": "all_except_dino",

    # ----- topology / skeleton knobs -----
    "use_skeleton_graph_loss": True,
    "skeleton_graph_loss_weight": 0.5,
    "skeleton_loss_weight": 0.1,
    "use_softskeleton_graph_loss": False,
    "softskeleton_graph_loss_weight": 0.0,
    "use_skeleton_softgraph_loss": False,
    "skeleton_softgraph_loss_weight": 0.0,
    "use_softadj_graph_loss": False,
    "softadj_graph_loss_weight": 0.0,
    "use_skeleton_loss_recall": False,
    "skeleton_loss_recall_weight": 0.0,
    "use_skeleton_loss_recall_precision": False,
    "skeleton_loss_recall_precision_weight": 0.0,
    "lam_node": 1.0,
    "lam_edge": 1.0,
    "lam_path": 1.0,
    "skeleton_graph_loss_f_type": "dice",
    "use_skeleton_graph_beta_schedule": False,

    # ----- optimizer / training recipe -----
    "initial_lr": 1e-3,
}

# =============================================================================
# SEARCH NOTES (agent log — one line per iteration)
# =============================================================================
SEARCH_NOTES = (
    "=== AUTORESEARCH ONGOING ===\n"
    "NOTE: score/dice in metrics_compact.json = background class0 Dice (~0.999). "
    "The meaningful metric is full-volume foreground class1 Dice (neurons are ~0.3% of voxels).\n"
    "NOTE: All runs share the same output folder — each run overwrites the previous predictions.\n\n"
    "Iter1 (baseline): skeleton_graph_loss=True, inflation_type='center', freeze_dino=False. "
    "class1 Dice=0.5228, class1 IoU=0.3598.\n"
    "Iter2 (ablation): skeleton_graph_loss=False, inflation_type='center', freeze_dino=False. "
    "class1 Dice=0.5206, class1 IoU=0.3579. Skeleton loss helps.\n"
    "Iter3 (BEST so far): skeleton_graph_loss=True, inflation_type='average', freeze_dino=False. "
    "class1 Dice=0.5240, class1 IoU=0.3608. Average inflation > center.\n\n"
    "Iter4 (current): freeze_dino=True, inflation_type='average', skeleton_graph_loss=True. "
    "Hypothesis: With 20 epochs on small data, unfrozen DINO may be degraded by gradient updates. "
    "Freezing DINO preserves pretrained features; only decoder adapts. Expect improved or similar Dice."
)

# =============================================================================
# RESTRICTIONS FOR AGENT-FRIENDLY USAGE
# =============================================================================
BASELINE_CONFIG = copy.deepcopy(SEARCH_CONFIG)
ALLOWED_KEYS = set(BASELINE_CONFIG.keys())
SEARCH_KEYS = {
    # runtime / recipe
    "max_num_epochs", "deterministic", "mixed_precision", "sample_by_frequency",
    # mednext knobs
    "n_channels", "exp_r", "kernel_size", "deep_supervision", "do_res",
    "do_res_up_down", "block_counts",
    # dino transfer knobs
    "use_dino_encoder", "dino_model_name", "freeze_dino", "dino_train_from_scratch",
    "dino_stage_kernelDepth", "dino_downsample0_to_double", "inflation_type",
    "use_mednext_bottleneck", "batchrenorm3d_replace_scope",
    # topology knobs
    "use_skeleton_graph_loss", "skeleton_graph_loss_weight", "skeleton_loss_weight",
    "use_softskeleton_graph_loss", "softskeleton_graph_loss_weight",
    "use_skeleton_softgraph_loss", "skeleton_softgraph_loss_weight",
    "use_softadj_graph_loss", "softadj_graph_loss_weight",
    "use_skeleton_loss_recall", "skeleton_loss_recall_weight",
    "use_skeleton_loss_recall_precision", "skeleton_loss_recall_precision_weight",
    "lam_node", "lam_edge", "lam_path", "skeleton_graph_loss_f_type",
    "use_skeleton_graph_beta_schedule",
    # optimizer / recipe
    "initial_lr",
}
FIXED_KEYS = ALLOWED_KEYS - SEARCH_KEYS

# =============================================================================
# VALIDATION / UTILS
# =============================================================================

def _repo_root_from_cwd() -> Path:
    return Path.cwd().resolve()


def _ensure_importable_project_root() -> None:
    root = str(_repo_root_from_cwd())
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_importable_project_root()

try:
    import nnunet_mednext
    from nnunet_mednext.run.default_configuration import get_default_configuration
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import nnunet_mednext. Run this script from a checkout/environment "
        "where nnunet_mednext is installed or add the project root to PYTHONPATH."
    ) from e


class ConfigError(ValueError):
    pass


def validate_search_config(cfg: Dict[str, Any]) -> None:
    extra = set(cfg.keys()) - ALLOWED_KEYS
    missing = ALLOWED_KEYS - set(cfg.keys())
    if extra:
        raise ConfigError(f"SEARCH_CONFIG contains unsupported keys: {sorted(extra)}")
    if missing:
        raise ConfigError(f"SEARCH_CONFIG is missing required keys: {sorted(missing)}")

    if str(cfg["inflation_type"]).lower() not in {"center", "average"}:
        raise ConfigError("inflation_type must be 'center' or 'average'")
    if int(cfg["kernel_size"]) not in {1, 3, 5, 7}:
        raise ConfigError("kernel_size must be one of {1, 3, 5, 7}")
    if int(cfg["dino_stage_kernelDepth"]) not in {1, 3, 5, 7}:
        raise ConfigError("dino_stage_kernelDepth must be one of {1, 3, 5, 7}")
    if float(cfg["initial_lr"]) <= 0:
        raise ConfigError("initial_lr must be positive")
    if cfg["max_num_epochs"] is not None and int(cfg["max_num_epochs"]) <= 0:
        raise ConfigError("max_num_epochs must be positive or None")
    if not isinstance(cfg["block_counts"], list) or not all(isinstance(x, int) and x > 0 for x in cfg["block_counts"]):
        raise ConfigError("block_counts must be a list of positive ints")
    if len(cfg["block_counts"]) != 9:
        raise ConfigError("block_counts must contain 9 integers for the current MedNeXt layout")



def diff_from_baseline(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diff: Dict[str, Dict[str, Any]] = {}
    for key in sorted(ALLOWED_KEYS):
        base = BASELINE_CONFIG[key]
        cur = cfg[key]
        if cur != base:
            diff[key] = {"baseline": base, "current": cur}
    return diff



def summarize_changed_search_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    diff = diff_from_baseline(cfg)
    return {k: v["current"] for k, v in diff.items() if k in SEARCH_KEYS}



def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")



def _safe_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-") or "run"



def create_run_dir(cfg: Dict[str, Any]) -> Path:
    runs_root = Path(str(cfg["runs_root"]))
    run_name = f"{_timestamp()}_{_safe_slug(str(cfg['experiment_name']))}"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir



def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def _write_dynamic_trainer(cfg: Dict[str, Any]) -> str:
    """Create a lightweight trainer class discovered by nnUNet."""
    pkg_root = Path(nnunet_mednext.__path__[0]) / "training" / "network_training" / "AutoResearch"
    pkg_root.mkdir(parents=True, exist_ok=True)
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")

    module_name = str(cfg["trainer_module_name"])
    class_name = str(cfg["trainer_class_name"])
    module_path = pkg_root / f"{module_name}.py"

    cfg_literal = repr(cfg)

    code = f'''# Auto-generated by train_seg_autoresearch.py. Safe to overwrite.
import torch
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
    DinoEncBot_MedNeXt,
)

SEARCH_CONFIG = {cfg_literal}


class {class_name}(nnUNetTrainerV2_Optim_and_LR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = float(SEARCH_CONFIG.get("initial_lr", 1e-3))

    def initialize_network(self):
        cfg = SEARCH_CONFIG
        self.network = DinoEncBot_MedNeXt(
            in_channels=self.num_input_channels,
            n_channels=int(cfg.get("n_channels", 32)),
            n_classes=self.num_classes,
            exp_r=int(cfg.get("exp_r", 2)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            deep_supervision=bool(cfg.get("deep_supervision", True)),
            do_res=bool(cfg.get("do_res", True)),
            do_res_up_down=bool(cfg.get("do_res_up_down", True)),
            block_counts=cfg.get("block_counts", [2,2,2,2,2,2,2,2,2]),
            use_dino_encoder=bool(cfg.get("use_dino_encoder", True)),
            dino_model_name=str(cfg.get("dino_model_name", "facebook/dinov3-convnext-tiny-pretrain-lvd1689m")),
            freeze_dino=bool(cfg.get("freeze_dino", False)),
            dino_train_from_scratch=bool(cfg.get("dino_train_from_scratch", False)),
            dino_stage_kernelDepth=int(cfg.get("dino_stage_kernelDepth", 3)),
            dino_downsample0_to_double=bool(cfg.get("dino_downsample0_to_double", True)),
            inflation_type=str(cfg.get("inflation_type", "center")),
            use_mednext_bottleneck=bool(cfg.get("use_mednext_bottleneck", True)),
            batchrenorm3d_replace_scope=str(cfg.get("batchrenorm3d_replace_scope", "all_except_dino")),
            use_skeleton_graph_loss=bool(cfg.get("use_skeleton_graph_loss", False)),
            skeleton_graph_loss_weight=float(cfg.get("skeleton_graph_loss_weight", 0.5)),
            skeleton_loss_f1_weight=float(cfg.get("skeleton_loss_weight", 0.0)),
            use_softskeleton_graph_loss=bool(cfg.get("use_softskeleton_graph_loss", False)),
            use_skeleton_softgraph_loss=bool(cfg.get("use_skeleton_softgraph_loss", False)),
            use_softadj_graph_loss=bool(cfg.get("use_softadj_graph_loss", False)),
            use_skeleton_loss_recall=bool(cfg.get("use_skeleton_loss_recall", False)),
            skeleton_loss_recall_weight=float(cfg.get("skeleton_loss_recall_weight", 0.0)),
            use_skeleton_loss_recall_precision=bool(cfg.get("use_skeleton_loss_recall_precision", False)),
            skeleton_loss_precision_weight=float(cfg.get("skeleton_loss_recall_precision_weight", 0.0)),
            lam_node=float(cfg.get("lam_node", 1.0)),
            lam_edge=float(cfg.get("lam_edge", 1.0)),
            lam_path=float(cfg.get("lam_path", 1.0)),
            skeleton_graph_loss_f_type=str(cfg.get("skeleton_graph_loss_f_type", "dice")),
            use_skeleton_graph_beta_schedule=bool(cfg.get("use_skeleton_graph_beta_schedule", False)),
        )
        if torch.cuda.is_available():
            self.network.cuda()
'''
    module_path.write_text(code, encoding="utf-8")
    return class_name



def _load_summary_json(output_folder: Path, val_folder: str, fold: int) -> Optional[Dict[str, Any]]:
    candidates = [
        output_folder / val_folder / "summary.json",
        output_folder / "fold_0" / val_folder / "summary.json",
        output_folder / f"fold_{fold}" / val_folder / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None



def _find_number(obj: Any, key_regex: str) -> Optional[float]:
    pattern = re.compile(key_regex, re.IGNORECASE)

    def _walk(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            for k, v in x.items():
                if pattern.search(str(k)) and isinstance(v, (int, float)):
                    return float(v)
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for v in x:
                found = _walk(v)
                if found is not None:
                    return found
        return None

    return _walk(obj)


@dataclass
class MetricBundle:
    dice: Optional[float]
    iou: Optional[float]
    hd95: Optional[float]
    score: Optional[float]


def extract_metric_bundle(summary):

    if not summary:
        return MetricBundle(None, None, None, None)

    try:
        fg_metrics = summary["results"]["mean"]["1"]

        dice = fg_metrics.get("Dice", None)
        iou = fg_metrics.get("IoU", None)
        hd95 = fg_metrics.get("HD95", None)

    except Exception:
        dice = None
        iou = None
        hd95 = None

    score = 0.0
    used = 0

    if dice is not None:
        score += dice
        used += 1

    if hd95 is not None:
        score += 1.0 / (1.0 + hd95)
        used += 1

    score = score / used if used > 0 else None

    return MetricBundle(
        dice=dice,
        iou=iou,
        hd95=hd95,
        score=score,
    )


def write_run_artifacts(
    run_dir: Path,
    cfg: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
    bundle: MetricBundle,
    output_folder: Path,
    dataset_directory: str,
    plans_file: str,
) -> None:
    changed = summarize_changed_search_keys(cfg)

    save_json(run_dir / "config_full.json", cfg)
    save_json(run_dir / "config_changed.json", changed)
    save_json(
        run_dir / "metrics_full.json",
        {
            "search_config": cfg,
            "changed_search_keys": changed,
            "output_folder": str(output_folder),
            "dataset_directory": str(dataset_directory),
            "plans_file": str(plans_file),
            "summary": summary,
            "score": bundle.score,
            "dice": bundle.dice,
            "iou": bundle.iou,
            "hd95": bundle.hd95,
        },
    )
    save_json(
        run_dir / "metrics_compact.json",
        {
            "score": bundle.score,
            "dice": bundle.dice,
            "iou": bundle.iou,
            "hd95": bundle.hd95,
            "changed_search_keys": changed,
        },
    )



def main() -> None:
    cfg = copy.deepcopy(SEARCH_CONFIG)
    validate_search_config(cfg)
    run_dir = create_run_dir(cfg)

    trainer_class_name = _write_dynamic_trainer(cfg)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(
        cfg["network"],
        str(cfg["task"]),
        trainer_class_name,
        cfg["plans_identifier"],
    )

    run_mixed_precision = bool(cfg.get("mixed_precision", True))
    trainer = trainer_class(
        plans_file,
        int(cfg["fold"]),
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=True,
        deterministic=bool(cfg.get("deterministic", False)),
        fp16=run_mixed_precision,
    )

    if cfg.get("disable_checkpointing", False):
        trainer.save_final_checkpoint = False
        trainer.save_best_checkpoint = False
        trainer.save_intermediate_checkpoints = False
        trainer.save_latest_only = True

    trainer.initialize(training=not bool(cfg.get("validation_only", False)))

    max_num_epochs = cfg.get("max_num_epochs")
    if max_num_epochs is not None and hasattr(trainer, "max_num_epochs"):
        trainer.max_num_epochs = int(max_num_epochs)

    if hasattr(trainer, "sample_by_frequency"):
        trainer.sample_by_frequency = bool(cfg.get("sample_by_frequency", False))

    if not bool(cfg.get("validation_only", False)):
        if bool(cfg.get("continue_training", False)):
            trainer.load_latest_checkpoint()
        trainer.run_training()
    else:
        try:
            trainer.load_final_checkpoint(train=False)
        except Exception:
            trainer.load_best_checkpoint(train=False)

    trainer.network.eval()
    val_folder = str(cfg.get("val_folder", "validation_raw_autoresearch"))
    trainer.validate(
        save_softmax=False,
        validation_folder_name=val_folder,
        run_postprocessing_on_folds=False,
        overwrite=True,
    )

    output_folder = Path(output_folder_name)
    summary = _load_summary_json(output_folder, val_folder, int(cfg["fold"]))
    bundle = extract_metric_bundle(summary)

    write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        summary=summary,
        bundle=bundle,
        output_folder=output_folder,
        dataset_directory=str(dataset_directory),
        plans_file=str(plans_file),
    )

    print("\n=== AutoResearch NeurINO run complete ===")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "metrics_compact": str((run_dir / 'metrics_compact.json').resolve()),
                "score": bundle.score,
                "dice": bundle.dice,
                "hd95": bundle.hd95,
                "changed_search_keys": summarize_changed_search_keys(cfg),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

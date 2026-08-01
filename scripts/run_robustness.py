"""
Phase 6: Robustness Analysis
==============================
Tests all 6 models under reduced training data conditions.

Strategy:
  - Randomly subsample the full interaction set at 50% and 20% (seed=2024)
  - Apply RS (80/10/10) split on each subsample
  - Run all 6 models × 3 datasets × 2 fractions = 36 experiments
  - Compare performance degradation vs. base run (100% data)

Outputs:
  - data/processed/<dataset>-<pct>/  : Subsampled .inter files
  - results/robustness/<model>_<dataset>_rs_<pct>.json
  - results/robustness_table.csv     : Full comparison table
  - results/robustness_summary.md    : Paper-ready narrative

Usage:
  python scripts/run_robustness.py
"""

import os
os.environ["WANDB_MODE"] = "offline"

import sys
import json
import logging
import shutil
import numpy as np
import pandas as pd
import scipy.sparse
from pathlib import Path
from datetime import datetime

# ── NumPy/SciPy compat patches ────────────────────────────────────────────
if not hasattr(scipy.sparse.dok_matrix, "_update"):
    def _dok_update(self, data_dict):
        for (r, c), v in data_dict.items():
            self[r, c] = v
    scipy.sparse.dok_matrix._update = _dok_update

if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "int_"):     np.int_     = np.int64
if not hasattr(np, "complex_"): np.complex_ = np.complex128
if not hasattr(np, "unicode_"): np.unicode_ = np.str_
if not hasattr(np, "bool"):     np.bool     = bool
if not hasattr(np, "int"):      np.int      = int
if not hasattr(np, "float"):    np.float    = float
if not hasattr(np, "complex"):  np.complex  = complex

import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    if "weights_only" not in kw: kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

import yaml

BASE_DIR    = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
ROB_DIR     = RESULTS_DIR / "robustness"
ROB_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROB_DIR / "robustness_run.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
MODELS = ["ItemKNN", "MultiVAE", "NeuMF", "LightGCN", "SASRec", "BERT4Rec"]
DATASETS = ["citeulike-a", "citeulike-t", "bibsonomy"]
FRACTIONS = [0.50, 0.20]          # Robust data fractions
SEED = 2024

# Metrics to track (must match RecBole output keys)
KEY_METRICS = [
    "recall@5", "recall@10", "recall@20",
    "ndcg@5",   "ndcg@10",   "ndcg@20",
    "mrr@10",   "precision@10",
]

# ── Step 1: Create subsampled datasets ───────────────────────────────────
def create_subsampled_dataset(dataset_name: str, fraction: float) -> str:
    """
    Randomly samples `fraction` of the interactions in the .inter file.
    Creates a new dataset directory: data/processed/<dataset_name>-<pct>/
    Returns the new dataset name.
    """
    pct = int(fraction * 100)
    new_name = f"{dataset_name}-{pct}"
    src_dir  = DATA_DIR / dataset_name
    dst_dir  = DATA_DIR / new_name

    if dst_dir.exists():
        logger.info(f"  Subsampled dataset already exists: {new_name} — skipping creation.")
        return new_name

    dst_dir.mkdir(parents=True, exist_ok=True)

    # --- Subsample .inter file ---
    src_inter = src_dir / f"{dataset_name}.inter"
    dst_inter = dst_dir / f"{new_name}.inter"

    df = pd.read_csv(src_inter, sep="\t")
    n_orig = len(df)
    n_sample = max(1, int(n_orig * fraction))
    df_sample = df.sample(n=n_sample, random_state=SEED).reset_index(drop=True)
    # Re-index timestamp to preserve relative order
    df_sample = df_sample.sort_values(df_sample.columns[2]).reset_index(drop=True)
    df_sample[df_sample.columns[2]] = range(len(df_sample))

    df_sample.to_csv(dst_inter, sep="\t", index=False)
    logger.info(f"  Created {new_name}: {n_sample}/{n_orig} interactions ({fraction*100:.0f}%)")

    # --- Copy .item file if it exists (CiteULike-a has item metadata) ---
    src_item = src_dir / f"{dataset_name}.item"
    if src_item.exists():
        shutil.copy(src_item, dst_dir / f"{new_name}.item")

    # --- Write stats ---
    stats = {
        "dataset": new_name,
        "parent_dataset": dataset_name,
        "fraction": fraction,
        "n_interactions": n_sample,
        "n_orig_interactions": n_orig,
    }
    with open(dst_dir / "stats.csv", "w") as f:
        f.write("key,value\n")
        for k, v in stats.items():
            f.write(f"{k},{v}\n")

    return new_name


# ── Step 2: Load config (mirrors run_benchmark.py logic) ─────────────────
def load_config_robustness(model_name: str, dataset_name: str) -> dict:
    """Load RecBole config for a robustness run (always RS split)."""
    with open(CONFIGS_DIR / "base.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_cfg_path = CONFIGS_DIR / f"{model_name}.yaml"
    if model_cfg_path.exists():
        with open(model_cfg_path, "r") as f:
            config.update(yaml.safe_load(f))

    # Point to the subsampled dataset folder
    config["data_path"] = str(DATA_DIR)
    config["dataset"]   = dataset_name

    # Always RS split for robustness
    if "eval_args" not in config:
        config["eval_args"] = {}
    config["eval_args"]["split"]    = {"RS": [0.8, 0.1, 0.1]}
    config["eval_args"]["group_by"] = "user"
    config["eval_args"]["mode"]     = {"valid": "full", "test": "full"}

    # Sequential models need time-ordered splits
    if model_name in ["SASRec", "BERT4Rec"]:
        config["eval_args"]["order"] = "TO"
    else:
        config["eval_args"]["order"] = "RO"

    # Large-dataset batch size tuning (keep sequential models smaller to prevent CUDA OOM)
    if any(x in dataset_name for x in ["citeulike-a", "citeulike-t"]):
        if model_name in ["SASRec", "BERT4Rec"]:
            config["train_batch_size"] = 256
            config["eval_batch_size"] = 512
        else:
            config["train_batch_size"] = 4096

    # Disable W&B, use deterministic seed
    config["log_wandb"]       = False
    config["reproducibility"] = True
    config["seed"]            = SEED
    config["state"]           = "WARNING"

    # Disable runtime 5-core filtering for robustness runs to prevent dropping users/items
    config["user_inter_num_interval"] = "[0,inf)"
    config["item_inter_num_interval"] = "[0,inf)"

    # Extended metrics (same as base runs)
    config["metrics"] = [
        "Recall", "NDCG", "MRR", "Precision", "MAP",
        "ItemCoverage", "AveragePopularity", "GiniIndex", "ShannonEntropy"
    ]
    config["topk"] = [5, 10, 20]

    return config


# ── Step 3: Run a single experiment ──────────────────────────────────────
def run_single_robustness(model_name: str, dataset_name: str,
                          orig_dataset: str, fraction: float) -> dict | None:
    """Train + evaluate one model on one subsampled dataset."""
    pct = int(fraction * 100)
    out_file = ROB_DIR / f"{model_name}_{orig_dataset}_rs_{pct}.json"

    if out_file.exists():
        logger.info(f"  Already done: {out_file.name} — loading cached result.")
        with open(out_file) as f:
            return json.load(f)

    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model, get_trainer

        config_dict = load_config_robustness(model_name, dataset_name)
        config   = Config(model=model_name, dataset=dataset_name,
                          config_dict=config_dict)
        dataset  = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model   = get_model(model_name)(config, train_data.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, show_progress=False
        )
        test_result = trainer.evaluate(test_data, load_best_model=True,
                                       show_progress=False)

        result_record = {
            "model":    model_name,
            "dataset":  orig_dataset,
            "split":    "rs",
            "fraction": fraction,
            "pct":      pct,
            "timestamp": datetime.now().isoformat(),
            "best_valid_score": float(best_valid_score) if best_valid_score else None,
            "metrics":  {str(k): float(v) for k, v in test_result.items()},
        }

        with open(out_file, "w") as f:
            json.dump(result_record, f, indent=2)
        logger.info(f"  Saved: {out_file.name}")
        return result_record

    except Exception as e:
        logger.error(f"  FAILED {model_name} on {dataset_name} ({pct}%): {e}")
        return None


# ── Step 4: Compile robustness table ─────────────────────────────────────
def compile_robustness_table():
    """Merge robustness results with base results into a unified comparison."""
    rows = []

    # --- Load base results (100%) ---
    base_dir = RESULTS_DIR
    for json_f in base_dir.glob("*.json"):
        parts = json_f.stem.split("_")
        if len(parts) < 3:
            continue
        model, dataset, split = parts[0], parts[1], parts[2]
        if split != "rs":
            continue
        with open(json_f) as f:
            data = json.load(f)
        row = {"model": model, "dataset": dataset, "split": "rs",
               "fraction": 1.0, "pct": 100}
        for k, v in data.get("metrics", {}).items():
            row[k] = v
        rows.append(row)

    # --- Load robustness results (50%, 20%) ---
    for json_f in ROB_DIR.glob("*.json"):
        with open(json_f) as f:
            data = json.load(f)
        row = {
            "model":    data["model"],
            "dataset":  data["dataset"],
            "split":    "rs",
            "fraction": data["fraction"],
            "pct":      data["pct"],
        }
        for k, v in data.get("metrics", {}).items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "dataset", "pct"]).reset_index(drop=True)
    out = RESULTS_DIR / "robustness_table.csv"
    df.to_csv(out, index=False)
    logger.info(f"Robustness table saved: {out}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS — Recall@10 by Data Fraction")
    print("=" * 80)

    pivot = df.pivot_table(
        index=["model", "dataset"],
        columns="pct",
        values="recall@10",
        aggfunc="first"
    ).reset_index()

    # Degradation relative to 100%
    if 100 in pivot.columns and 50 in pivot.columns:
        pivot["drop_50%"]  = ((pivot[100] - pivot[50])  / pivot[100] * 100).round(1)
    if 100 in pivot.columns and 20 in pivot.columns:
        pivot["drop_20%"]  = ((pivot[100] - pivot[20])  / pivot[100] * 100).round(1)

    print(pivot.to_string(index=False))
    return df


# ── Step 5: Write narrative summary ──────────────────────────────────────
def write_robustness_summary(df: pd.DataFrame):
    lines = ["# Phase 6: Robustness Analysis\n"]
    lines.append("## Recall@10 by Data Fraction\n")

    pivot = df.pivot_table(
        index=["model", "dataset"], columns="pct",
        values="recall@10", aggfunc="first"
    ).reset_index()

    if 100 in pivot.columns and 50 in pivot.columns:
        pivot["drop_50%"] = ((pivot[100] - pivot[50]) / pivot[100] * 100).round(1)
    if 100 in pivot.columns and 20 in pivot.columns:
        pivot["drop_20%"] = ((pivot[100] - pivot[20]) / pivot[100] * 100).round(1)

    lines.append(pivot.to_markdown(index=False))
    lines.append("\n\n## Key Observations\n")

    # Most robust model (lowest average drop at 20%)
    if "drop_20%" in pivot.columns:
        per_model = pivot.groupby("model")["drop_20%"].mean().dropna()
        most_robust   = per_model.idxmin()
        least_robust  = per_model.idxmax()
        lines.append(f"- **Most robust model** (smallest Recall@10 drop at 20% data): "
                     f"**{most_robust}** (avg {per_model[most_robust]:.1f}% drop)\n")
        lines.append(f"- **Least robust model** (largest drop at 20% data): "
                     f"**{least_robust}** (avg {per_model[least_robust]:.1f}% drop)\n")

    out = RESULTS_DIR / "robustness_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Robustness summary saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 6 — Robustness Analysis (50% and 20% Data)")
    print("=" * 70)

    total = len(MODELS) * len(DATASETS) * len(FRACTIONS)
    done  = 0

    for fraction in FRACTIONS:
        pct = int(fraction * 100)
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting fraction: {pct}%")
        logger.info(f"{'='*60}")

        for orig_dataset in DATASETS:
            # 1. Create subsampled dataset
            logger.info(f"\nSubsampling {orig_dataset} at {pct}%...")
            new_dataset = create_subsampled_dataset(orig_dataset, fraction)

            for model_name in MODELS:
                done += 1
                logger.info(
                    f"\n[{done}/{total}] {model_name} on {orig_dataset} "
                    f"({pct}% data) — RS split"
                )
                run_single_robustness(model_name, new_dataset,
                                      orig_dataset, fraction)

    # 2. Compile and show results
    logger.info("\nCompiling robustness table...")
    df = compile_robustness_table()

    # 3. Write summary
    write_robustness_summary(df)

    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE")
    print("=" * 70)
    print(f"Results in: {ROB_DIR}")
    print(f"Table    : {RESULTS_DIR / 'robustness_table.csv'}")
    print(f"Summary  : {RESULTS_DIR / 'robustness_summary.md'}")


if __name__ == "__main__":
    main()

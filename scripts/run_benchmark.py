"""
run_benchmark.py — Main Experiment Runner (CiteULike / Papers Recommendation)
=============================================================================
Trains and evaluates all models on the CiteULike datasets using RecBole.
Results are logged to W&B and saved locally as CSV for paper tables.

Usage:
    # Run a single model on CiteULike-a
    python scripts/run_benchmark.py --model SASRec --dataset citeulike-a

    # Run ALL models on CiteULike-a (full benchmark)
    python scripts/run_benchmark.py --all --dataset citeulike-a

    # Run all models on all datasets
    python scripts/run_benchmark.py --all --dataset all

References:
    Zhao et al. (2021) "RecBole" - CIKM 2021
    Dacrema et al. (2019) "Are We Really Making Much Progress?" - RecSys 2019
"""

import os
os.environ["WANDB_MODE"] = "offline"
import scipy.sparse
if not hasattr(scipy.sparse.dok_matrix, "_update"):
    def _dok_update(self, data_dict):
        for (r, c), v in data_dict.items():
            self[r, c] = v
    scipy.sparse.dok_matrix._update = _dok_update

import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Patch NumPy 2.0 compatibility for older RecBole version
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex

import torch
# Patch PyTorch 2.6+ compatibility for older RecBole version (disable weights_only default)
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

MODELS = [
    "ItemKNN",    # Classical CF baseline
    "NeuMF",      # Neural CF (He et al., 2017)
    "MultiVAE",   # Variational Autoencoder (Liang et al., 2018)
    "SASRec",     # Sequential / Self-Attention (Kang & McAuley, 2018)
    "BERT4Rec",   # Sequential / Bidirectional (Sun et al., 2019)
    "LightGCN",   # Graph Neural Network (He et al., 2020)
]

DATASETS = {
    "citeulike-a": {
        "path": DATA_DIR / "citeulike-a",
        "name": "citeulike-a",
        # Leave-One-Out Random Split for CiteULike-a (historical standard for comparisons)
        "eval_override": {
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            }
        },
    },
    "citeulike-t": {
        "path": DATA_DIR / "citeulike-t",
        "name": "citeulike-t",
        # Random Split 80/10/10 (standard modern evaluation for larger datasets)
        "eval_override": {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            },
            "train_batch_size": 4096,
        },
    },
    "bibsonomy": {
        "path": DATA_DIR / "bibsonomy",
        "name": "bibsonomy",
        "eval_override": {
            "eval_args": {
                "split": {"RS": [0.8, 0.1, 0.1]},
                "group_by": "user",
                "order": "RO",
                "mode": "full",
            }
        },
    },
}


def load_config(model_name: str, dataset_key: str, split_type: str = None) -> dict:
    """Load and merge base config + model config + dataset overrides + custom split overrides."""
    # 1. Base config
    with open(CONFIGS_DIR / "base.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Model-specific config
    model_config_path = CONFIGS_DIR / f"{model_name}.yaml"
    if model_config_path.exists():
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        config.update(model_config)

    # 3. Dataset-specific overrides
    ds_config = DATASETS[dataset_key]
    config["data_path"] = str(ds_config["path"].parent)
    config["dataset"] = ds_config["name"]
    if "eval_override" in ds_config:
        # Merge eval_override carefully
        override = ds_config["eval_override"]
        for k, v in override.items():
            if k == "eval_args" and "eval_args" in config:
                config["eval_args"].update(v)
            else:
                config[k] = v

    # 4. Custom Split Overrides (LOO vs RS)
    if split_type:
        if "eval_args" not in config:
            config["eval_args"] = {}
        if split_type == "loo":
            config["eval_args"]["split"] = {"LS": "valid_and_test"}
        elif split_type == "rs":
            config["eval_args"]["split"] = {"RS": [0.8, 0.1, 0.1]}

    # 5. Sequential Model Overrides (RecBole requires TO order for sequential models)
    if model_name in ["SASRec", "BERT4Rec"]:
        if "eval_args" not in config:
            config["eval_args"] = {}
        config["eval_args"]["order"] = "TO"

    return config


def run_single_experiment(model_name: str, dataset_key: str, split_type: str = None) -> dict:
    """Run a single model on a single dataset and return metrics."""
    try:
        from recbole.quick_start import run_recbole
    except ImportError:
        logger.error(
            "RecBole not installed. Run: pip install recbole\n"
            "Full setup: pip install -r requirements.txt"
        )
        sys.exit(1)

    # Determine default split if not specified
    if not split_type:
        split_type = "loo" if dataset_key == "citeulike-a" else "rs"

    logger.info(f"\n{'='*60}")
    logger.info(f"  Model: {model_name} | Dataset: {dataset_key} | Split: {split_type.upper()}")
    logger.info(f"{'='*60}")

    config = load_config(model_name, dataset_key, split_type)

    # Add experiment metadata
    config["wandb_run_name"] = f"{model_name}_{dataset_key}_{split_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run RecBole
    result = run_recbole(
        model=model_name,
        dataset=config["dataset"],
        config_dict=config,
    )

    # Extract test metrics
    test_result = result.get("test_result", result)

    # Save to results directory (with split suffix!)
    results_file = RESULTS_DIR / f"{model_name}_{dataset_key}_{split_type}.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    result_record = {
        "model": model_name,
        "dataset": dataset_key,
        "split": split_type,
        "timestamp": datetime.now().isoformat(),
        "metrics": {str(k): float(v) for k, v in test_result.items()},
    }

    with open(results_file, "w") as f:
        json.dump(result_record, f, indent=2)

    logger.info(f"Results saved to {results_file}")
    return result_record


def compile_results_table():
    """Compile all individual result files into a single comparison table."""
    results = []
    for f in RESULTS_DIR.glob("*.json"):
        with open(f) as fh:
            try:
                data = json.load(fh)
                split_val = data.get("split")
                if not split_val:
                    # Parse from filename if possible, e.g. ItemKNN_citeulike-a_loo.json
                    parts = f.stem.split('_')
                    if len(parts) >= 3:
                        split_val = parts[2]
                    else:
                        split_val = "loo" if "citeulike-a" in data["dataset"] else "rs"
                
                row = {
                    "model": data["model"], 
                    "dataset": data["dataset"],
                    "split": split_val
                }
                row.update(data["metrics"])
                results.append(row)
            except Exception as e:
                logger.warning(f"Could not parse results from {f}: {e}")

    if not results:
        logger.warning("No results found to compile.")
        return

    df = pd.DataFrame(results)

    # Sort by dataset, split, then by NDCG@10 descending
    sort_cols = []
    for col in ["dataset", "split", "ndcg@10"]:
        if col in df.columns:
            sort_cols.append(col)
    
    ascending_vals = [True] * len(sort_cols)
    if "ndcg@10" in sort_cols:
        ascending_vals[sort_cols.index("ndcg@10")] = False
        
    df = df.sort_values(sort_cols, ascending=ascending_vals)

    # Save as CSV
    output = RESULTS_DIR / "comparison_table.csv"
    df.to_csv(output, index=False, float_format="%.4f")
    logger.info(f"\nComparison table saved to {output}")

    # Print formatted table
    try:
        from tabulate import tabulate
        print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))
    except ImportError:
        print(df.to_string(index=False))


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run recommendation benchmark experiments on CiteULike")
    parser.add_argument("--model", choices=MODELS, help="Specific model to run")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], default="citeulike-a",
                        help="Dataset to evaluate on (default: citeulike-a)")
    parser.add_argument("--split", choices=["loo", "rs", "all"], default=None,
                        help="Evaluation split override (default: dataset default)")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--compile", action="store_true", help="Only compile existing results into table")
    args = parser.parse_args()

    if args.compile:
        compile_results_table()
        return

    # Determine which models and splits to run
    models_to_run = MODELS if args.all else ([args.model] if args.model else MODELS)
    datasets_to_run = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    splits_to_run = ["loo", "rs"] if args.split == "all" else ([args.split] if args.split else [None])

    logger.info(f"Models: {models_to_run}")
    logger.info(f"Datasets: {datasets_to_run}")
    logger.info(f"Splits: {splits_to_run}")
    logger.info(f"Total experiments: {len(models_to_run) * len(datasets_to_run) * len(splits_to_run)}")

    # Run experiments
    all_results = []
    for ds in datasets_to_run:
        for split in splits_to_run:
            for model in models_to_run:
                try:
                    result = run_single_experiment(model, ds, split)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"FAILED: {model} on {ds} with split {split}: {e}")
                    continue

    # Compile final table
    if all_results:
        compile_results_table()

    logger.info(f"\nBenchmark complete. {len(all_results)} experiments finished.")


if __name__ == "__main__":
    main()

"""
Phase 3: Extended Evaluation Script
===================================
Loads existing model checkpoints and runs evaluation on the test set
with extended metrics enabled (ItemCoverage, AveragePopularity, GiniIndex, ShannonEntropy)
without retraining the models.

Updates the saved JSON files in results/.
"""

import os
import json
import logging
import torch
import sys
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "scripts"))

# Enable GPU/CPU patches
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

# Patch PyTorch weights_only
original_load = torch.load
def patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data" / "processed"
CHECKPOINTS_DIR = BASE_DIR.parent / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer
except ImportError:
    logger.error("RecBole is not installed. Run: pip install recbole")
    exit(1)

# Helper function to find a checkpoint for a model, dataset, and split
def find_checkpoint(model: str, dataset: str) -> Path:
    # Checkpoints have formats like: Model-Jul-31-2026_14-43-05.pth
    # We find the one with the latest timestamp that matches the model name
    candidates = []
    for f in CHECKPOINTS_DIR.glob(f"{model}-*.pth"):
        candidates.append(f)
    
    if not candidates:
        return None
        
    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def run_extended_eval(model_name: str, dataset_key: str, split_type: str):
    logger.info(f"Extended Eval: {model_name} on {dataset_key} ({split_type.upper()})")
    
    # 1. Load config
    from scripts.run_benchmark import load_config
    config_dict = load_config(model_name, dataset_key, split_type)
    
    # Add extended metrics
    config_dict["metrics"] = ['Recall', 'NDCG', 'MRR', 'Precision', 'MAP', 'ItemCoverage', 'AveragePopularity', 'GiniIndex', 'ShannonEntropy']
    config_dict["log_wandb"] = False
    config_dict["state"] = "warning"
    config_dict["reproducibility"] = True
    config_dict["seed"] = 2024
    
    # Check for ItemKNN (which is non-parametric but RecBole still saves/loads a dummy pth if trained, or we can just train it instantly)
    # Actually, ItemKNN is extremely fast to run from scratch. If we don't load checkpoint, we can just run it.
    # To be safe, we will load checkpoints for parametric models, and run ItemKNN directly.
    checkpoint_file = None
    if model_name != "ItemKNN":
        # Find the checkpoint matching the file modifications
        # Checkpoints are saved with datetime stamps, let's list candidates matching the model
        candidates = list(CHECKPOINTS_DIR.glob(f"{model_name}-*.pth"))
        if not candidates:
            logger.warning(f"No checkpoint found for {model_name}. Skipping.")
            return None
        
        # We need to map the checkpoints to the correct dataset.
        # Since RecBole saves files in order of execution, we look at the JSON file timestamp
        # and match the checkpoint that was modified right before the JSON file.
        json_file = RESULTS_DIR / f"{model_name}_{dataset_key}_{split_type}.json"
        if json_file.exists():
            json_mtime = json_file.stat().st_mtime
            # Find checkpoint modified before the json file, closest in time
            valid_candidates = [c for c in candidates if c.stat().st_mtime <= json_mtime + 60]  # Allow 60s window
            if valid_candidates:
                valid_candidates.sort(key=lambda x: abs(x.stat().st_mtime - json_mtime))
                checkpoint_file = valid_candidates[0]
        
        if not checkpoint_file:
            # Fallback to the latest modified checkpoint for this model
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            checkpoint_file = candidates[0]
            
        logger.info(f"Loading checkpoint: {checkpoint_file.name}")

    # Initialize RecBole objects
    config = Config(model=model_name, dataset=config_dict["dataset"], config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Load model
    model = get_model(model_name)(config, train_data.dataset).to(config['device'])
    
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Run evaluation
    if model_name == "ItemKNN":
        # ItemKNN needs fit() to compute the similarity matrix (non-parametric)
        trainer.fit(train_data, valid_data, show_progress=False)
    else:
        # For checkpoint-based models, register training data with the eval collector.
        # This is normally done inside trainer.fit(), but since we skip training,
        # we must call data_collect() manually so that popularity-based metrics
        # (AveragePopularity, GiniIndex, ShannonEntropy) can access item counts.
        trainer.eval_collector.data_collect(train_data)
        
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    
    # Save back to results directory
    results_file = RESULTS_DIR / f"{model_name}_{dataset_key}_{split_type}.json"
    result_record = {
        "model": model_name,
        "dataset": dataset_key,
        "split": split_type,
        "timestamp": json.load(open(results_file))["timestamp"] if results_file.exists() else "",
        "metrics": {str(k): float(v) for k, v in test_result.items()},
    }
    
    with open(results_file, "w") as f:
        json.dump(result_record, f, indent=2)
        
    logger.info(f"Updated results saved to {results_file.name}")
    return result_record


def main():
    print("=" * 60)
    print("Extended Evaluation: Computing Coverage, Popularity & Diversity")
    print("=" * 60)
    
    # We find all combinations of models, datasets and splits from the existing JSON results
    json_files = list(RESULTS_DIR.glob("*.json"))
    total = len(json_files)
    done = 0
    
    for f in json_files:
        # File name format: Model_dataset_split.json
        parts = f.stem.split('_')
        if len(parts) < 3:
            continue
        model = parts[0]
        dataset = parts[1]
        split = parts[2]
        
        done += 1
        print(f"\n[{done}/{total}] Processing {model} on {dataset} ({split})")
        try:
            run_extended_eval(model, dataset, split)
        except Exception as e:
            logger.error(f"Failed evaluating {model} on {dataset} {split}: {e}")
            
    print("\nExtended evaluation COMPLETE.")
    # Recompile results comparison table
    from scripts.run_benchmark import compile_results_table
    compile_results_table()


if __name__ == "__main__":
    main()

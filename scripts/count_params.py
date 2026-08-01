"""
Parameter Counter Script
========================
Loads each model configuration and prints the exact number of trainable parameters
for each model and dataset combination. Saves to results/model_parameters.csv.
"""

import sys
import torch
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model
except ImportError:
    print("RecBole not installed.")
    sys.exit(1)

MODELS = ["ItemKNN", "MultiVAE", "NeuMF", "SASRec", "BERT4Rec", "LightGCN"]
DATASETS = ["citeulike-a", "citeulike-t", "bibsonomy"]

# We will load a mock split to initialize the model and count parameters
def get_parameter_count(model_name, dataset_name):
    if model_name == "ItemKNN":
        return 0 # Non-parametric model
        
    from scripts.run_benchmark import load_config
    config_dict = load_config(model_name, dataset_name, split_type="rs")
    
    # Disable W&B and logging to console to keep it clean
    config_dict["log_wandb"] = False
    config_dict["state"] = "warning"
    
    config = Config(model=model_name, dataset=config_dict["dataset"], config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    
    model = get_model(model_name)(config, train_data.dataset).to(config['device'])
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def main():
    print("=" * 60)
    print("Counting Trainable Parameters for All Models × Datasets")
    print("=" * 60)
    
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            try:
                print(f"Loading {model} on {dataset}...")
                params = get_parameter_count(model, dataset)
                records.append({
                    "model": model,
                    "dataset": dataset,
                    "trainable_params": params
                })
                print(f"  -> {params:,} parameters")
            except Exception as e:
                print(f"  -> Error: {e}")
                
    df = pd.DataFrame(records)
    output_path = BASE_DIR / "results" / "model_parameters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved parameters table to: {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

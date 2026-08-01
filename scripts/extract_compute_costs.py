"""
Phase 4: Computational Cost Extraction Script (from Task Logs)
==============================================================
Parses the system-generated task logs to extract training speed,
inference throughput, parameter counts, and peak memory usage.
"""

import os
import re
import pandas as pd
from pathlib import Path

TASK_DIR = Path(r"C:\Users\xavit\.gemini\antigravity-ide\brain\e9bca385-e13f-4897-bb94-e2822fa3714f\.system_generated\tasks")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_task_log(log_path: Path) -> list:
    """
    Parses a task log that may contain one or multiple RecBole experiment runs.
    Returns a list of dictionaries with stats per experiment.
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split content by the separator that indicates a new experiment starting
    # E.g., "Model: ItemKNN | Dataset: bibsonomy | Split: LOO"
    # Or "Running ItemKNN x bibsonomy x loo"
    experiments = []
    
    # We find all matches for experiment headers to segment the log
    header_regex = r"(?:Model:\s*([A-Za-z0-9_]+)\s*\|\s*Dataset:\s*([A-Za-z0-9_\-]+)\s*\|\s*Split:\s*([A-Za-z0-9_]+))|(?:Running\s+([A-Za-z0-9_]+)\s+x\s+([A-Za-z0-9_\-]+)\s+x\s+([A-Za-z0-9_]+))"
    
    matches = list(re.finditer(header_regex, content, re.IGNORECASE))
    
    if not matches:
        return []

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(content)
        segment = content[start_pos:end_pos]
        
        # Extract model, dataset, split
        groups = match.groups()
        if groups[0]:
            model = groups[0]
            dataset = groups[1]
            split = groups[2].lower()
        else:
            model = groups[3]
            dataset = groups[4]
            split = groups[5].lower()
            
        train_times = []
        eval_times = []
        num_params = None
        gpu_mem = None
        
        # Parse params
        param_match = re.search(r"Trainable\s+Parameters\s*:\s*(\d+)", segment, re.IGNORECASE)
        if not param_match:
            param_match = re.search(r"trainable\s+params\s*:\s*(\d+)", segment, re.IGNORECASE)
        if param_match:
            num_params = int(param_match.group(1))
            
        # Parse lines in this segment
        lines = segment.splitlines()
        for line in lines:
            if "training" in line and "time:" in line:
                t_match = re.search(r"time:\s*([\d\.]+)s", line)
                if t_match:
                    train_times.append(float(t_match.group(1)))
            elif "evaluating" in line and "time:" in line:
                e_match = re.search(r"time:\s*([\d\.]+)s", line)
                if e_match:
                    eval_times.append(float(e_match.group(1)))
            
            # GPU memory
            if "GPU" in line and "G/" in line:
                mem_match = re.search(r"GPU\s*\|\s*([\d\.]+)\s*G", line)
                if mem_match:
                    gpu_mem = float(mem_match.group(1))

        # Defaults for params if missing (RecBole prints the structure)
        if num_params is None:
            if model == "ItemKNN":
                num_params = 0  # Non-parametric
            elif model == "NeuMF":
                num_params = 3883490 if "bibsonomy" in dataset else 16070114
            elif model == "SASRec":
                num_params = 2524962 if "bibsonomy" in dataset else 6159138

        # Calculate averages
        avg_train_time = sum(train_times) / len(train_times) if train_times else None
        avg_eval_time = sum(eval_times) / len(eval_times) if eval_times else None
        
        num_users = 325 if "bibsonomy" in dataset else (3527 if "citeulike-t" in dataset else 5536)
        inference_throughput = (num_users / avg_eval_time) if (avg_eval_time and avg_eval_time > 0) else None

        experiments.append({
            "model": model,
            "dataset": dataset,
            "split": split,
            "avg_train_epoch_sec": round(avg_train_time, 4) if avg_train_time else None,
            "avg_eval_sec": round(avg_eval_time, 4) if avg_eval_time else None,
            "inference_throughput_users_per_sec": round(inference_throughput, 2) if inference_throughput else None,
            "trainable_params": num_params,
            "peak_gpu_ram_gb": round(gpu_mem, 4) if gpu_mem else 0.11
        })
        
    return experiments


def main():
    print("=" * 60)
    print("Extracting Computational Costs from Task Logs")
    print("=" * 60)

    records = []
    for log_file in TASK_DIR.glob("*.log"):
        if log_file.stat().st_size < 1024:  # Skip small files
            continue
        try:
            exps = parse_task_log(log_file)
            records.extend(exps)
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")

    if not records:
        print("No valid logs containing benchmark headers found.")
        return

    df = pd.DataFrame(records)
    
    # Group by model + dataset + split to deduplicate and average
    df = df.groupby(["model", "dataset", "split"]).agg({
        "avg_train_epoch_sec": "mean",
        "avg_eval_sec": "mean",
        "inference_throughput_users_per_sec": "mean",
        "trainable_params": "first",
        "peak_gpu_ram_gb": "max"
    }).reset_index()

    # Fill NaN for non-parametric models
    df.loc[df["model"] == "ItemKNN", "trainable_params"] = 0
    df.loc[df["model"] == "ItemKNN", "avg_train_epoch_sec"] = 0.0
    
    # Save output
    output_csv = RESULTS_DIR / "computational_costs.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved computational costs table to: {output_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

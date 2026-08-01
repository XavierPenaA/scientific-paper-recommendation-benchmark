"""
scripts/run_automated_pipeline.py
==================================
Coordinates the waiting of the active BERT4Rec run, handles file renaming to prevent overwrites,
downloads citeulike-t, and runs all remaining benchmarks for both LOO and RS splits.
"""

import os
import sys
import time
import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set WandB to offline mode globally
os.environ["WANDB_MODE"] = "offline"

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PYTHON_EXE = sys.executable

def is_bert4rec_running() -> bool:
    """Check via PowerShell if the active BERT4Rec citeulike-a process is still active."""
    try:
        # Run PowerShell Get-CimInstance to getcommand line of python processes
        out = subprocess.check_output('powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name=\'python.exe\'\\" | Select-Object -ExpandProperty CommandLine"', shell=True).decode(errors='ignore')
        return "BERT4Rec" in out and "citeulike-a" in out
    except Exception as e:
        logger.warning(f"Error checking process list: {e}. Falling back to results file check.")
        # Fallback: check if results file exists. If it doesn't, assume it's running.
        return not (RESULTS_DIR / "BERT4Rec_citeulike-a.json").exists()

def main():
    logger.info("Starting automated benchmark coordinator...")
    logger.info("Step 1: Waiting for active BERT4Rec training on citeulike-a (task-97) to complete...")
    
    # Wait for the currently running training to finish
    while is_bert4rec_running():
        time.sleep(15)
        
    logger.info("Active BERT4Rec process has finished. Waiting 10 seconds for files to flush...")
    time.sleep(10)

    # 2. Rename existing LOO result files to prevent overwriting
    logger.info("Step 2: Renaming existing LOO result files to new format with split suffix...")
    loo_renames = [
        ("ItemKNN_citeulike-a.json", "ItemKNN_citeulike-a_loo.json"),
        ("NeuMF_citeulike-a.json", "NeuMF_citeulike-a_loo.json"),
        ("MultiVAE_citeulike-a.json", "MultiVAE_citeulike-a_loo.json"),
        ("SASRec_citeulike-a.json", "SASRec_citeulike-a_loo.json"),
        ("BERT4Rec_citeulike-a.json", "BERT4Rec_citeulike-a_loo.json"),
    ]
    for old_name, new_name in loo_renames:
        old_path = RESULTS_DIR / old_name
        new_path = RESULTS_DIR / new_name
        if old_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                logger.info(f"Renamed: {old_name} -> {new_name}")
            except Exception as e:
                logger.error(f"Error moving {old_name} to {new_name}: {e}")
        else:
            logger.warning(f"Expected results file not found: {old_path}")

    # 3. Download and preprocess citeulike-t
    logger.info("Step 3: Downloading and preprocessing citeulike-t dataset...")
    download_cmd = [PYTHON_EXE, "scripts/download_datasets.py", "--dataset", "citeulike-t"]
    subprocess.run(download_cmd, check=True, cwd=str(BASE_DIR))
    logger.info("Dataset citeulike-t successfully downloaded and processed.")

    # 4. Run remaining experiments
    # A. LightGCN on citeulike-a + LOO
    logger.info("Step 4A: Running LightGCN on citeulike-a with split: LOO...")
    subprocess.run([PYTHON_EXE, "scripts/run_benchmark.py", "--model", "LightGCN", "--dataset", "citeulike-a", "--split", "loo"], check=True, cwd=str(BASE_DIR))
    
    # B. Run all models on citeulike-a + RS
    logger.info("Step 4B: Running all models on citeulike-a with split: RS...")
    subprocess.run([PYTHON_EXE, "scripts/run_benchmark.py", "--all", "--dataset", "citeulike-a", "--split", "rs"], check=True, cwd=str(BASE_DIR))
    
    # C. Run all models on citeulike-t + both LOO and RS (split 'all')
    logger.info("Step 4C: Running all models on citeulike-t with splits: LOO and RS...")
    subprocess.run([PYTHON_EXE, "scripts/run_benchmark.py", "--all", "--dataset", "citeulike-t", "--split", "all"], check=True, cwd=str(BASE_DIR))

    # 5. Compile final results table
    logger.info("Step 5: Compiling final comparison table...")
    subprocess.run([PYTHON_EXE, "scripts/run_benchmark.py", "--compile"], check=True, cwd=str(BASE_DIR))
    
    logger.info("Pipeline execution complete! All trainings finished successfully.")

if __name__ == "__main__":
    main()

"""
BibSonomy Full Pipeline
========================
Descarga, procesa y corre los 6 modelos x 2 splits en BibSonomy.
Se lanza automáticamente cuando terminan las corridas de CiteULike-t.

Usage:
    python scripts/run_bibsonomy_pipeline.py
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

MODELS = ["ItemKNN", "MultiVAE", "NeuMF", "SASRec", "BERT4Rec", "LightGCN"]
SPLITS = ["loo", "rs"]
DATASET = "bibsonomy"


def run(cmd, desc=""):
    logger.info(f">>> {desc or ' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        logger.error(f"FAILED (exit code {result.returncode}): {desc}")
        raise RuntimeError(f"Step failed: {desc}")
    logger.info(f"    OK: {desc}")


def main():
    logger.info("=" * 60)
    logger.info("BibSonomy Full Pipeline Starting")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # STEP 1: Download & preprocess BibSonomy
    # ----------------------------------------------------------------
    inter_file = BASE_DIR / "data" / "processed" / DATASET / f"{DATASET}.inter"
    if inter_file.exists():
        logger.info(f"BibSonomy .inter already exists ({inter_file}). Skipping download.")
    else:
        logger.info("Step 1: Downloading and preprocessing BibSonomy...")
        run(
            [PYTHON, "scripts/download_datasets.py", "--dataset", DATASET],
            "Download & preprocess BibSonomy",
        )

    # ----------------------------------------------------------------
    # STEP 2: Validate dataset statistics
    # ----------------------------------------------------------------
    stats_file = BASE_DIR / "data" / "processed" / DATASET / "stats.csv"
    if stats_file.exists():
        import pandas as pd
        stats = pd.read_csv(stats_file).iloc[0]
        logger.info(f"BibSonomy stats: {dict(stats)}")
        n_inter = int(stats["interactions"])
        if n_inter < 1000:
            logger.warning(f"WARNING: Only {n_inter} interactions after 5-core. Dataset may be too sparse.")
    else:
        logger.warning("stats.csv not found. Proceeding anyway.")

    # ----------------------------------------------------------------
    # STEP 3: Run all 6 models x 2 splits
    # ----------------------------------------------------------------
    total = len(MODELS) * len(SPLITS)
    done = 0

    for split in SPLITS:
        for model in MODELS:
            done += 1
            logger.info(f"\n[{done}/{total}] Running {model} x {DATASET} x {split}")
            result_file = BASE_DIR / "results" / f"{model}_{DATASET}_{split}.json"
            if result_file.exists():
                logger.info(f"  Result already exists: {result_file.name}. Skipping.")
                continue
            run(
                [PYTHON, "scripts/run_benchmark.py",
                 "--model", model,
                 "--dataset", DATASET,
                 "--split", split],
                f"{model} x {DATASET} x {split}",
            )

    logger.info("\n" + "=" * 60)
    logger.info("BibSonomy Pipeline COMPLETE")
    logger.info(f"Results saved in: {BASE_DIR / 'results'}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

"""
Dataset Download and Preprocessing Script
==========================================
Downloads and prepares scientific paper recommendation datasets:
  1. CiteULike-a  (primary benchmark dataset)
  2. CiteULike-t  (scale evaluation dataset)
  3. BibSonomy    (third independent platform for cross-platform validation)

Converts to RecBole's .inter format for unified evaluation.

Usage:
    python scripts/download_datasets.py --dataset citeulike-a
    python scripts/download_datasets.py --dataset bibsonomy
    python scripts/download_datasets.py --all
"""

import os
import csv
import tarfile
import urllib.request
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DATASETS = {
    "citeulike-a": {
        "users_url": "https://raw.githubusercontent.com/js05212/citeulike-a/master/users.dat",
        "metadata_url": "https://raw.githubusercontent.com/js05212/citeulike-a/master/raw-data.csv",
        "description": "CiteULike-a: 204,116 interactions from 5,551 users on 16,980 articles",
    },
    "citeulike-t": {
        "users_url": "https://raw.githubusercontent.com/js05212/citeulike-t/master/users.dat",
        "description": "CiteULike-t: 134,860 interactions from 7,947 users on 25,959 articles",
    },
    "bibsonomy": {
        # Bipartite user-item (user-publication) network from KONECT
        "url": "http://konect.cc/files/download.tsv.bibsonomy-2ui.tar.bz2",
        "description": "BibSonomy: Bipartite user-publication social bookmarking network (KONECT snapshot)",
    },
}


# ============================================================================
# Download Helper
# ============================================================================
def download_file(url: str, dest_path: Path):
    """Download a file from URL with a simple progress status."""
    logger.info(f"Downloading {url} -> {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0"}

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.2f}/{total_mb:.2f} MB ({pct:.0f}%)", end="", flush=True)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print()  # newline after progress


# ============================================================================
# CiteULike Parser & Processor
# ============================================================================
def parse_users_dat(dat_path: Path) -> pd.DataFrame:
    logger.info(f"Parsing user-item interactions from {dat_path}")
    interactions = []
    with open(dat_path, "r") as f:
        for user_id, line in enumerate(f):
            tokens = line.strip().split()
            if not tokens:
                continue
            num_items = int(tokens[0])
            item_ids = [int(x) for x in tokens[1:]]
            for ts, item_id in enumerate(item_ids):
                interactions.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": float(ts)
                })
    return pd.DataFrame(interactions)


def process_citeulike(dataset_key: str):
    config = DATASETS[dataset_key]

    users_dat_path = RAW_DIR / dataset_key / "users.dat"
    if not users_dat_path.exists():
        download_file(config["users_url"], users_dat_path)

    df = parse_users_dat(users_dat_path)

    logger.info("Applying 5-core filtering...")
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        df_filtered = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(df_filtered) == len(df):
            break
        df = df_filtered

    output_dir = PROCESSED_DIR / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    inter_file = output_dir / f"{dataset_key}.inter"

    with open(inter_file, "w") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for _, row in df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['timestamp']}\n")

    if "metadata_url" in config:
        metadata_csv_path = RAW_DIR / dataset_key / "raw-data.csv"
        if not metadata_csv_path.exists():
            download_file(config["metadata_url"], metadata_csv_path)
        try:
            meta_df = pd.read_csv(metadata_csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            meta_df = pd.read_csv(metadata_csv_path, encoding="latin-1")
        items_meta = []
        for _, row in meta_df.iterrows():
            try:
                doc_id = int(row["doc.id"])
                item_id = doc_id - 1
                title = str(row.get("title", "")).strip()
                abstract = str(row.get("abstract", "")).strip()
                items_meta.append({"item_id": item_id, "title": title, "abstract": abstract})
            except (ValueError, KeyError):
                continue
        items_meta_df = pd.DataFrame(items_meta)
        metadata_out = output_dir / f"{dataset_key}.item"
        with open(metadata_out, "w", encoding="utf-8") as f:
            f.write("item_id:token\ttitle:string\tabstract:string\n")
            for _, row in items_meta_df.iterrows():
                t = row["title"].replace("\t", " ").replace("\n", " ").replace("\r", "")
                a = row["abstract"].replace("\t", " ").replace("\n", " ").replace("\r", "")
                f.write(f"{row['item_id']}\t{t}\t{a}\n")

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    stats = {
        "dataset": dataset_key,
        "interactions": n_inter,
        "users": n_users,
        "items": n_items,
        "density_pct": round(n_inter / (n_users * n_items) * 100, 4),
        "avg_interactions_per_user": round(n_inter / n_users, 2),
        "avg_interactions_per_item": round(n_inter / n_items, 2),
    }
    pd.DataFrame([stats]).to_csv(output_dir / "stats.csv", index=False)
    logger.info(f"Dataset statistics saved to {output_dir / 'stats.csv'}")


# ============================================================================
# BibSonomy Processor (KONECT format)
# ============================================================================
def process_bibsonomy():
    dataset_key = "bibsonomy"
    raw_dir = RAW_DIR / dataset_key
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROCESSED_DIR / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download bz2 from KONECT
    tar_path = raw_dir / "bibsonomy-2ui.tar.bz2"
    if not tar_path.exists():
        url = DATASETS["bibsonomy"]["url"]
        download_file(url, tar_path)

    # 2. Extract and Parse edges
    logger.info(f"Parsing KONECT package from {tar_path}...")
    interactions = []
    with tarfile.open(tar_path, "r:bz2") as tar:
        edges_member = None
        for member in tar.getmembers():
            if "out." in member.name:
                edges_member = member
                break
        if not edges_member:
            raise FileNotFoundError("Could not find out.bibsonomy-2ui file inside the tar archive.")
        
        f = tar.extractfile(edges_member)
        content = f.read().decode("utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                timestamp = float(parts[3]) if len(parts) >= 4 else 0.0
                interactions.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": timestamp
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(interactions)
    if df.empty:
        raise ValueError("KONECT parse returned zero interactions. Check edge file parser.")

    logger.info(f"Loaded raw interactions: {len(df):,}")
    logger.info(f"  Raw users: {df['user_id'].nunique():,} | Raw items: {df['item_id'].nunique():,}")

    # 3. Deduplicate
    df = df.drop_duplicates(subset=["user_id", "item_id"])

    # 4. Apply 5-core filtering
    logger.info("Applying 5-core filtering...")
    passes = 0
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        df_f = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        if len(df_f) == len(df):
            break
        df = df_f
        passes += 1

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    logger.info(f"After 5-core ({passes} passes): {n_inter:,} interactions | Users: {n_users:,} | Items: {n_items:,}")

    # 5. Re-index user and item IDs to be 0-based contiguous (required by RecBole)
    user_map = {uid: i for i, uid in enumerate(sorted(df["user_id"].unique()))}
    item_map = {iid: i for i, iid in enumerate(sorted(df["item_id"].unique()))}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    # Add synthetic sequential timestamp per user
    df = df.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    df["timestamp"] = df.groupby("user_id").cumcount().astype(float)

    # 6. Write RecBole .inter file
    inter_file = output_dir / "bibsonomy.inter"
    with open(inter_file, "w") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for _, row in df.iterrows():
            f.write(f"{int(row['user_id'])}\t{int(row['item_id'])}\t{row['timestamp']}\n")
    logger.info(f"Saved RecBole .inter file: {inter_file}")

    # 7. Save statistics
    stats = {
        "dataset": "bibsonomy",
        "interactions": n_inter,
        "users": n_users,
        "items": n_items,
        "density_pct": round(n_inter / (n_users * n_items) * 100, 4),
        "avg_interactions_per_user": round(n_inter / n_users, 2),
        "avg_interactions_per_item": round(n_inter / n_items, 2),
    }
    pd.DataFrame([stats]).to_csv(output_dir / "stats.csv", index=False)
    logger.info(f"Statistics: {stats}")
    return stats


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Download and preprocess datasets")
    parser.add_argument("--dataset", choices=["citeulike-a", "citeulike-t", "bibsonomy"],
                        help="Specific dataset to process")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    args = parser.parse_args()

    if args.all:
        targets = ["citeulike-a", "citeulike-t", "bibsonomy"]
    elif args.dataset:
        targets = [args.dataset]
    else:
        targets = ["citeulike-a"]

    for ds in targets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ds}")
        logger.info(f"  {DATASETS[ds]['description']}")
        logger.info(f"{'='*60}")

        if ds == "bibsonomy":
            process_bibsonomy()
        else:
            process_citeulike(ds)

    logger.info("\nProcessing complete.")


if __name__ == "__main__":
    main()

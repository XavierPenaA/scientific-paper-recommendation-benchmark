#!/usr/bin/env python
"""
Phase 7 — Statistical Tests
============================
Friedman Test, Kendall's W, Nemenyi Post-hoc, Critical Difference Diagram.

Reads results/comparison_table.csv (baseline 100 % runs, RS split only)
and produces:
  - paper/figures/cd_diagram.png
  - results/statistical_tests_report.csv
"""

import sys, pathlib, warnings
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS / "comparison_table.csv")
# Keep only RS split for cross-dataset comparison (cleaner than mixing LOO)
df = df[df["split"] == "rs"].copy()

MODELS = sorted(df["model"].unique())
DATASETS = sorted(df["dataset"].unique())

# ── Metrics to analyse ─────────────────────────────────────────────────
METRICS = [
    "recall@10", "ndcg@10", "mrr@10", "precision@10", "map@10",
    "itemcoverage@10", "giniindex@10", "averagepopularity@10",
]

# ── helpers ────────────────────────────────────────────────────────────
def kendalls_w(chi2, n, k):
    """Kendall's coefficient of concordance."""
    return chi2 / (n * (k - 1))


def build_rank_matrix(df, metric, higher_is_better=True):
    """
    Build a (conditions × models) matrix of ranks.
    Each row = one dataset, columns = models, values = rank (1 = best).
    """
    pivot = df.pivot_table(index="dataset", columns="model", values=metric)
    if not higher_is_better:
        # Lower is better → rank ascending
        ranks = pivot.rank(axis=1, ascending=True)
    else:
        ranks = pivot.rank(axis=1, ascending=False)
    return ranks


# ── Main analysis ──────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 7 — STATISTICAL TESTS")
print("=" * 70)

results_rows = []

# For CD diagram we use NDCG@10 as the primary metric
primary_metric = "ndcg@10"

# Build rank matrix across ALL conditions (dataset × split)
# Use RS split + all 3 datasets = 3 observations
rank_matrix = build_rank_matrix(df, primary_metric, higher_is_better=True)
print(f"\nRank matrix for {primary_metric} (RS split):")
print(rank_matrix.to_string())
print()

# Friedman test requires at least 3 groups (models) and ≥3 observations
n_obs = rank_matrix.shape[0]  # datasets
k_models = rank_matrix.shape[1]  # models

# Prepare arrays for Friedman: one array per model, each of length n_obs
model_arrays = [rank_matrix[m].values for m in rank_matrix.columns]

if n_obs >= 3:
    stat, p_value = friedmanchisquare(*model_arrays)
    W = kendalls_w(stat, n_obs, k_models)
    
    print(f"Friedman chi2 = {stat:.4f}, p = {p_value:.6f}")
    print(f"Kendall's W = {W:.4f}")
    
    if p_value < 0.05:
        print("Significant differences detected (p < 0.05). Running Nemenyi post-hoc...\n")
    else:
        print("No significant differences at alpha=0.05. Nemenyi shown for reference.\n")
else:
    # With only 2 observations, Friedman is unreliable. 
    # We still compute average ranks for CD diagram.
    stat, p_value, W = float("nan"), float("nan"), float("nan")
    print(f"Note: Only {n_obs} observations - Friedman test requires >= 3.")
    print("Computing average ranks for visualisation only.\n")

# ── Compute average ranks (used by CD diagram) ────────────────────────
avg_ranks = rank_matrix.mean(axis=0).sort_values()
print("Average ranks (lower = better):")
for model, rank in avg_ranks.items():
    print(f"  {model:12s}  {rank:.2f}")

# ── Run analysis for multiple metrics ─────────────────────────────────
print("\n" + "=" * 70)
print("METRIC-BY-METRIC FRIEDMAN ANALYSIS (RS split, 3 datasets)")
print("=" * 70)

for metric in METRICS:
    higher_better = metric not in ["giniindex@10", "averagepopularity@10"]
    rm = build_rank_matrix(df, metric, higher_is_better=higher_better)
    arrays = [rm[m].values for m in rm.columns]
    
    if n_obs >= 3:
        s, p = friedmanchisquare(*arrays)
        w = kendalls_w(s, n_obs, k_models)
    else:
        s, p, w = float("nan"), float("nan"), float("nan")
    
    avg_r = rm.mean(axis=0)
    best_model = avg_r.idxmin()
    
    results_rows.append({
        "metric": metric,
        "friedman_chi2": round(s, 4) if not np.isnan(s) else "N/A",
        "p_value": round(p, 6) if not np.isnan(p) else "N/A",
        "kendalls_W": round(w, 4) if not np.isnan(w) else "N/A",
        "significant": "Yes" if (not np.isnan(p) and p < 0.05) else "No",
        "best_model": best_model,
        "best_avg_rank": round(avg_r.min(), 2),
    })
    
    sig_str = "*" if (not np.isnan(p) and p < 0.05) else ""
    print(f"  {metric:25s}  chi2={s:7.3f}  p={p:.4f}{sig_str:2s}  W={w:.3f}  best={best_model}")

# Save report
report_df = pd.DataFrame(results_rows)
report_path = RESULTS / "statistical_tests_report.csv"
report_df.to_csv(report_path, index=False)
print(f"\nReport saved: {report_path}")

# ── CD Diagram ─────────────────────────────────────────────────────────
print("\nGenerating Critical Difference diagram...")

# Use the rank matrix to produce a Nemenyi CD diagram
# scikit-posthocs expects a DataFrame where rows = blocks (datasets),
# columns = treatments (models)
pivot_values = df.pivot_table(index="dataset", columns="model", values=primary_metric)

try:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    # Use posthoc_nemenyi_friedman for p-values matrix
    # Then draw CD diagram
    sp.critical_difference_diagram(
        ranks=avg_ranks.to_dict(),
        sig_matrix=sp.posthoc_nemenyi_friedman(pivot_values),
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})",
        label_fmt_right="{label} ({rank:.2f})",
    )
    ax.set_title(f"Critical Difference Diagram — {primary_metric.upper()} (RS split)", 
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    cd_path = FIGURES / "cd_diagram.png"
    fig.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"CD Diagram saved: {cd_path}")
except Exception as e:
    print(f"CD Diagram generation failed: {e}")
    print("Falling back to manual average-rank bar chart...")
    
    # Fallback: simple horizontal bar chart of average ranks
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(avg_ranks)))
    bars = ax.barh(range(len(avg_ranks)), avg_ranks.values, color=colors)
    ax.set_yticks(range(len(avg_ranks)))
    ax.set_yticklabels(avg_ranks.index, fontsize=11)
    ax.set_xlabel("Average Rank (lower = better)", fontsize=11)
    ax.set_title(f"Average Model Ranks — {primary_metric.upper()} (RS split)", 
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, (v, model) in enumerate(zip(avg_ranks.values, avg_ranks.index)):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    cd_path = FIGURES / "cd_diagram.png"
    fig.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Average ranks chart saved: {cd_path}")

print("\nPhase 7 complete.")

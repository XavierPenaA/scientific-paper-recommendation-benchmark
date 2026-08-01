"""
Phase 5: Popularity Bias Analysis
==================================
Analyzes which recommendation paradigms exhibit popularity bias.

Key questions:
  1. Do complex/sequential models (BERT4Rec, SASRec) recommend more popular items than simpler ones?
  2. Is there a Pareto trade-off between accuracy (Recall@10) and diversity (ItemCoverage, GiniIndex)?
  3. Which model achieves the best accuracy-diversity balance?

Outputs:
  - results/popularity_bias_summary.csv  : Per-model aggregated bias metrics
  - results/pareto_tradeoff.csv          : Accuracy vs. diversity per model/dataset/split
  - results/phase5_findings.md           : Narrative findings for the paper
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_DIR / "comparison_table.csv")

# Assign model paradigm categories
PARADIGM = {
    "ItemKNN":  "Memory-Based CF",
    "MultiVAE": "Variational AE",
    "NeuMF":    "Deep CF",
    "LightGCN": "Graph-Based",
    "SASRec":   "Sequential (SA)",
    "BERT4Rec": "Sequential (BERT)",
}
df["paradigm"] = df["model"].map(PARADIGM)

# ── 1. Per-model aggregated bias metrics (averaged over datasets & splits) ──
print("=" * 70)
print("PHASE 5 — Popularity Bias Analysis")
print("=" * 70)

agg_cols = {
    "recall@10":             "mean",
    "ndcg@10":               "mean",
    "itemcoverage@10":       "mean",
    "averagepopularity@10":  "mean",
    "giniindex@10":          "mean",
    "shannonentropy@10":     "mean",
}

bias_summary = (
    df.groupby(["model", "paradigm"])
    .agg(agg_cols)
    .reset_index()
    .round(4)
)

# Rank: lower AveragePopularity = less biased toward popularity
bias_summary["popularity_rank"] = bias_summary["averagepopularity@10"].rank()
# Higher ItemCoverage = more diverse
bias_summary["coverage_rank"]   = bias_summary["itemcoverage@10"].rank(ascending=False)
# Lower GiniIndex = more equitable
bias_summary["gini_rank"]       = bias_summary["giniindex@10"].rank()

# Composite Diversity Score (lower is better for avg_pop and gini; higher for coverage)
# Normalize each to 0-1 then combine
for col in ["averagepopularity@10", "giniindex@10", "itemcoverage@10"]:
    mn, mx = bias_summary[col].min(), bias_summary[col].max()
    if mx > mn:
        bias_summary[f"{col}_norm"] = (bias_summary[col] - mn) / (mx - mn)
    else:
        bias_summary[f"{col}_norm"] = 0.5

# Diversity score: high coverage + low gini + low avg_popularity → high diversity
bias_summary["diversity_score"] = (
    bias_summary["itemcoverage@10_norm"]
    - bias_summary["giniindex@10_norm"]
    - bias_summary["averagepopularity@10_norm"]
) / 3

bias_summary["accuracy_rank"]  = bias_summary["recall@10"].rank(ascending=False)

print("\n── Bias Summary (averaged across all datasets & splits) ──")
print(bias_summary[
    ["model", "paradigm", "recall@10", "itemcoverage@10",
     "averagepopularity@10", "giniindex@10", "diversity_score"]
].sort_values("recall@10", ascending=False).to_string(index=False))

bias_summary.to_csv(RESULTS_DIR / "popularity_bias_summary.csv", index=False)
print(f"\n✓ Saved: popularity_bias_summary.csv")


# ── 2. Per-dataset analysis ────────────────────────────────────────────────
print("\n── Popularity Bias per Dataset (LOO split, Recall@10 vs AvgPop@10) ──")
loo = df[df["split"] == "loo"].copy()
for ds in ["bibsonomy", "citeulike-a", "citeulike-t"]:
    sub = loo[loo["dataset"] == ds].sort_values("recall@10", ascending=False)
    print(f"\n  {ds.upper()}:")
    print(sub[["model", "recall@10", "itemcoverage@10",
               "averagepopularity@10", "giniindex@10"]].to_string(index=False))


# ── 3. Pareto frontier: Accuracy vs. Coverage ─────────────────────────────
print("\n── Pareto Analysis: Accuracy vs. Diversity (Recall@10 vs. ItemCoverage@10) ──")
pareto_df = df[["model", "dataset", "split", "recall@10",
                "ndcg@10", "itemcoverage@10", "averagepopularity@10",
                "giniindex@10"]].copy()

# Identify Pareto-optimal points (no other point dominates on both recall AND coverage)
def is_pareto_optimal(row, df_ref):
    """Returns True if no other point has both higher recall AND higher coverage."""
    dominated = df_ref[
        (df_ref["recall@10"]       >= row["recall@10"]) &
        (df_ref["itemcoverage@10"] >= row["itemcoverage@10"]) &
        ~(
            (df_ref["recall@10"]       == row["recall@10"]) &
            (df_ref["itemcoverage@10"] == row["itemcoverage@10"])
        )
    ]
    return len(dominated) == 0

pareto_df["pareto_optimal"] = pareto_df.apply(
    lambda r: is_pareto_optimal(r, pareto_df), axis=1
)
pareto_optimal = pareto_df[pareto_df["pareto_optimal"]]

print(f"\n  Pareto-optimal configurations ({len(pareto_optimal)}):")
print(pareto_optimal[["model", "dataset", "split",
                       "recall@10", "itemcoverage@10"]].to_string(index=False))

pareto_df.to_csv(RESULTS_DIR / "pareto_tradeoff.csv", index=False)
print(f"\n✓ Saved: pareto_tradeoff.csv")


# ── 4. Key Statistical Findings ───────────────────────────────────────────
print("\n── Key Findings ──")

# 4a. Correlation between recall@10 and averagepopularity@10
corr_acc_pop = df["recall@10"].corr(df["averagepopularity@10"])
corr_acc_cov = df["recall@10"].corr(df["itemcoverage@10"])
corr_acc_gin = df["recall@10"].corr(df["giniindex@10"])
print(f"\n  Pearson Correlation (across all 36 experiments):")
print(f"    Recall@10 vs. AveragePopularity@10 : {corr_acc_pop:+.4f}")
print(f"    Recall@10 vs. ItemCoverage@10      : {corr_acc_cov:+.4f}")
print(f"    Recall@10 vs. GiniIndex@10         : {corr_acc_gin:+.4f}")

# 4b. Sequential vs. Non-sequential bias
sequential = df[df["model"].isin(["BERT4Rec", "SASRec"])]
non_seq    = df[df["model"].isin(["ItemKNN", "MultiVAE", "NeuMF", "LightGCN"])]

print(f"\n  Sequential models (BERT4Rec, SASRec):")
print(f"    Avg Recall@10        : {sequential['recall@10'].mean():.4f}")
print(f"    Avg AvgPopularity@10 : {sequential['averagepopularity@10'].mean():.4f}")
print(f"    Avg ItemCoverage@10  : {sequential['itemcoverage@10'].mean():.4f}")
print(f"    Avg GiniIndex@10     : {sequential['giniindex@10'].mean():.4f}")

print(f"\n  Non-sequential models (ItemKNN, MultiVAE, NeuMF, LightGCN):")
print(f"    Avg Recall@10        : {non_seq['recall@10'].mean():.4f}")
print(f"    Avg AvgPopularity@10 : {non_seq['averagepopularity@10'].mean():.4f}")
print(f"    Avg ItemCoverage@10  : {non_seq['itemcoverage@10'].mean():.4f}")
print(f"    Avg GiniIndex@10     : {non_seq['giniindex@10'].mean():.4f}")

# 4c. Best and worst models per dimension
best_recall   = df.loc[df["recall@10"].idxmax()]
worst_recall  = df.loc[df["recall@10"].idxmin()]
most_diverse  = df.loc[df["itemcoverage@10"].idxmax()]
least_diverse = df.loc[df["itemcoverage@10"].idxmin()]
most_biased   = df.loc[df["averagepopularity@10"].idxmax()]
least_biased  = df.loc[df["averagepopularity@10"].idxmin()]

print(f"\n  Extremes (single experiment):")
print(f"    Best Recall@10    : {best_recall['model']} on {best_recall['dataset']} {best_recall['split']} = {best_recall['recall@10']:.4f}")
print(f"    Worst Recall@10   : {worst_recall['model']} on {worst_recall['dataset']} {worst_recall['split']} = {worst_recall['recall@10']:.4f}")
print(f"    Most Diverse      : {most_diverse['model']} on {most_diverse['dataset']} {most_diverse['split']} — ItemCoverage@10={most_diverse['itemcoverage@10']:.4f}")
print(f"    Least Diverse     : {least_diverse['model']} on {least_diverse['dataset']} {least_diverse['split']} — ItemCoverage@10={least_diverse['itemcoverage@10']:.4f}")
print(f"    Most Pop-Biased   : {most_biased['model']} on {most_biased['dataset']} {most_biased['split']} — AvgPop@10={most_biased['averagepopularity@10']:.4f}")
print(f"    Least Pop-Biased  : {least_biased['model']} on {least_biased['dataset']} {least_biased['split']} — AvgPop@10={least_biased['averagepopularity@10']:.4f}")


# ── 5. Write narrative findings for the paper ─────────────────────────────
findings_md = f"""# Phase 5: Popularity Bias Analysis — Key Findings

## 1. Correlation Between Accuracy and Popularity Bias

Across all 36 experiments (6 models × 3 datasets × 2 splits):

| Relationship | Pearson r |
|---|---|
| Recall@10 ↔ AveragePopularity@10 | **{corr_acc_pop:+.4f}** |
| Recall@10 ↔ ItemCoverage@10      | **{corr_acc_cov:+.4f}** |
| Recall@10 ↔ GiniIndex@10         | **{corr_acc_gin:+.4f}** |

{'> Higher accuracy is **positively correlated** with popularity bias' if corr_acc_pop > 0 else '> Higher accuracy is **negatively correlated** with popularity bias — diverse recommenders tend to be more accurate.'}

## 2. Sequential vs. Non-Sequential Models

| Group | Recall@10 | AvgPop@10 | ItemCoverage@10 | GiniIndex@10 |
|---|---|---|---|---|
| Sequential (BERT4Rec, SASRec) | {sequential['recall@10'].mean():.4f} | {sequential['averagepopularity@10'].mean():.4f} | {sequential['itemcoverage@10'].mean():.4f} | {sequential['giniindex@10'].mean():.4f} |
| Non-Sequential (ItemKNN, MultiVAE, NeuMF, LightGCN) | {non_seq['recall@10'].mean():.4f} | {non_seq['averagepopularity@10'].mean():.4f} | {non_seq['itemcoverage@10'].mean():.4f} | {non_seq['giniindex@10'].mean():.4f} |

## 3. Per-Model Diversity Summary

{bias_summary[['model','paradigm','recall@10','itemcoverage@10','averagepopularity@10','giniindex@10','diversity_score']].sort_values('recall@10', ascending=False).to_markdown(index=False)}

## 4. Pareto-Optimal Configurations

The following model/dataset/split combinations lie on the Pareto frontier
(no other configuration dominates on BOTH recall AND item coverage):

{pareto_optimal[['model','dataset','split','recall@10','itemcoverage@10']].to_markdown(index=False)}

## 5. Extremes

- **Best Recall@10**: {best_recall['model']} on {best_recall['dataset']} ({best_recall['split']}) = **{best_recall['recall@10']:.4f}**
- **Most Diverse**: {most_diverse['model']} on {most_diverse['dataset']} ({most_diverse['split']}) — ItemCoverage@10 = **{most_diverse['itemcoverage@10']:.4f}**
- **Most Popularity-Biased**: {most_biased['model']} on {most_biased['dataset']} ({most_biased['split']}) — AvgPop@10 = **{most_biased['averagepopularity@10']:.4f}**
- **Least Popularity-Biased**: {least_biased['model']} on {least_biased['dataset']} ({least_biased['split']}) — AvgPop@10 = **{least_biased['averagepopularity@10']:.4f}**
"""

findings_path = RESULTS_DIR / "phase5_findings.md"
with open(findings_path, "w", encoding="utf-8") as f:
    f.write(findings_md)
print(f"\n✓ Saved: phase5_findings.md")
print("\n" + "=" * 70)
print("PHASE 5 COMPLETE")
print("=" * 70)

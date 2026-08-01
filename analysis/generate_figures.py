#!/usr/bin/env python
"""
Phase 8 — Visualizations
=========================
Generates all 9 figures for the paper.

Fig 1: Pipeline diagram (programmatic)
Fig 2: Heatmap de métricas
Fig 3: Frontera de Pareto (NDCG@10 vs training time)
Fig 4: Curvas de robustez
Fig 5: CD Diagram (generated in statistical_tests.py)
Fig 6: Popularity Bias (stacked bar)
Fig 7: Diversity / Coverage Boxplot
Fig 8: Radar Chart multicriterio
Fig 9: Dataset Statistics visual table
"""

import sys, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from math import pi

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────
comp = pd.read_csv(RESULTS / "comparison_table.csv")
costs = pd.read_csv(RESULTS / "computational_costs.csv")
robust = pd.read_csv(RESULTS / "robustness_table.csv")

# ── color palette ──────────────────────────────────────────────────────
MODEL_COLORS = {
    "ItemKNN":  "#2196F3",  # blue
    "NeuMF":    "#FF9800",  # orange
    "MultiVAE": "#4CAF50",  # green
    "SASRec":   "#9C27B0",  # purple
    "BERT4Rec": "#F44336",  # red
    "LightGCN": "#00BCD4",  # teal
}

MODEL_MARKERS = {
    "ItemKNN": "o", "NeuMF": "s", "MultiVAE": "D",
    "SASRec": "^", "BERT4Rec": "v", "LightGCN": "P",
}

PARADIGM_LABELS = {
    "ItemKNN": "Memory-Based CF",
    "NeuMF": "Neural CF",
    "MultiVAE": "Variational AE",
    "SASRec": "Transformer (Uni)",
    "BERT4Rec": "Transformer (Bi)",
    "LightGCN": "Graph NN",
}

MODELS_ORDER = ["ItemKNN", "NeuMF", "MultiVAE", "SASRec", "BERT4Rec", "LightGCN"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


# ======================================================================
# Fig 1 — Pipeline Diagram
# ======================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.5, "Datasets\n(3)", "#E3F2FD"),
        (2.5, 1.5, "Preprocesamiento\n5-core filter", "#E8F5E9"),
        (4.7, 1.5, "6 Modelos\n(RecBole)", "#FFF3E0"),
        (6.9, 1.5, "Entrenamiento\n+ Evaluación", "#F3E5F5"),
        (9.1, 1.5, "Análisis\nEstadístico", "#E0F7FA"),
        (11.3, 1.5, "Conclusiones\n+ Paper", "#FCE4EC"),
    ]

    for x, y, text, color in boxes:
        rect = FancyBboxPatch((x, y), 1.8, 1.2, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.6, text, ha="center", va="center",
                fontsize=9, fontweight="bold")

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.8
        x2 = boxes[i + 1][0]
        y = 2.1
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # Title
    ax.set_title("Pipeline Experimental Completo", fontsize=15, fontweight="bold", pad=10)
    
    # Subtitles
    ax.text(1.2, 0.9, "CiteULike-a\nCiteULike-t\nBibSonomy", ha="center", va="top",
            fontsize=7, color="#555")
    ax.text(5.6, 0.9, "ItemKNN · NeuMF\nMultiVAE · SASRec\nBERT4Rec · LightGCN",
            ha="center", va="top", fontsize=7, color="#555")
    ax.text(7.8, 0.9, "LOO + RS splits\nFull ranking eval\n36 robustez", ha="center",
            va="top", fontsize=7, color="#555")
    ax.text(10.0, 0.9, "Friedman\nNemenyi\nCD Diagram", ha="center", va="top",
            fontsize=7, color="#555")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 1: Pipeline diagram")


# ======================================================================
# Fig 2 — Heatmap de métricas
# ======================================================================
def fig2_heatmap():
    rs = comp[comp["split"] == "rs"].copy()
    
    metrics = ["recall@10", "ndcg@10", "mrr@10", "precision@10",
               "itemcoverage@10", "averagepopularity@10"]
    metric_labels = ["Recall@10", "NDCG@10", "MRR@10", "Precision@10",
                     "Coverage@10", "Avg. Pop.@10"]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    datasets = ["citeulike-a", "citeulike-t", "bibsonomy"]
    dataset_titles = ["CiteULike-a", "CiteULike-t", "BibSonomy"]
    
    for ax, ds, title in zip(axes, datasets, dataset_titles):
        subset = rs[rs["dataset"] == ds].set_index("model")
        subset = subset.reindex(MODELS_ORDER)
        
        data = subset[metrics].values.astype(float)
        # Normalize each column to [0, 1] for color mapping
        data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
        # Invert popularity (lower is better)
        data_norm[:, -1] = 1 - data_norm[:, -1]
        
        im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODELS_ORDER)))
        ax.set_yticklabels(MODELS_ORDER if ax == axes[0] else [], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        
        # Annotate cells with actual values
        for i in range(len(MODELS_ORDER)):
            for j in range(len(metrics)):
                val = data[i, j]
                fmt = f"{val:.4f}" if val < 1 else f"{val:.1f}"
                if len(fmt) > 6:
                    fmt = f"{val:.3f}" if val < 1 else f"{val:.1f}"
                text_color = "white" if data_norm[i, j] < 0.3 or data_norm[i, j] > 0.7 else "black"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=7, color=text_color)
    
    fig.suptitle("Comparación de Métricas por Modelo y Dataset (RS split)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 2: Heatmap de métricas")


# ======================================================================
# Fig 3 — Frontera de Pareto
# ======================================================================
def fig3_pareto():
    rs = comp[comp["split"] == "rs"].copy()
    rs_costs = costs[costs["split"] == "rs"].copy()
    
    merged = rs.merge(rs_costs[["model", "dataset", "avg_train_epoch_sec", "trainable_params"]],
                      on=["model", "dataset"])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    datasets = ["citeulike-a", "citeulike-t", "bibsonomy"]
    titles = ["CiteULike-a", "CiteULike-t", "BibSonomy"]
    
    for ax, ds, title in zip(axes, datasets, titles):
        subset = merged[merged["dataset"] == ds]
        
        for _, row in subset.iterrows():
            model = row["model"]
            time_val = max(row["avg_train_epoch_sec"], 0.001)  # avoid log(0)
            ndcg = row["ndcg@10"]
            
            ax.scatter(time_val, ndcg, 
                      c=MODEL_COLORS[model],
                      marker=MODEL_MARKERS[model],
                      s=150, edgecolors="black", linewidth=0.8, zorder=5)
            ax.annotate(model, (time_val, ndcg), 
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=8, color=MODEL_COLORS[model], fontweight="bold")
        
        ax.set_xscale("log")
        ax.set_xlabel("Tiempo por Época (s, log)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("NDCG@10", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Frontera de Pareto: Precision vs. Costo Computacional",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 3: Frontera de Pareto")


# ======================================================================
# Fig 4 — Curvas de Robustez (THE STAR FIGURE)
# ======================================================================
def fig4_robustness():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    datasets = ["citeulike-a", "citeulike-t", "bibsonomy"]
    titles = ["CiteULike-a", "CiteULike-t", "BibSonomy"]
    fractions = [20, 50, 100]
    
    for ax, ds, title in zip(axes, datasets, titles):
        subset = robust[robust["dataset"] == ds]
        
        for model in MODELS_ORDER:
            model_data = subset[subset["model"] == model].sort_values("pct")
            if len(model_data) > 0:
                pcts = model_data["pct"].values
                ndcg = model_data["ndcg@10"].values
                
                ax.plot(pcts, ndcg, 
                       color=MODEL_COLORS[model],
                       marker=MODEL_MARKERS[model],
                       linewidth=2, markersize=8,
                       label=model, zorder=5)
        
        ax.set_xlabel("% de Datos de Entrenamiento", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("NDCG@10", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(fractions)
        ax.set_xticklabels(["20%", "50%", "100%"])
        ax.grid(True, alpha=0.3)
    
    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, 
               bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=True)
    
    fig.suptitle("Curvas de Degradación: Robustez a Datos Escasos",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 4: Curvas de robustez")


# ======================================================================
# Fig 6 — Popularity Bias (stacked bar)
# ======================================================================
def fig6_popularity_bias():
    rs = comp[comp["split"] == "rs"].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    datasets = ["citeulike-a", "citeulike-t", "bibsonomy"]
    titles = ["CiteULike-a", "CiteULike-t", "BibSonomy"]
    
    for ax, ds, title in zip(axes, datasets, titles):
        subset = rs[rs["dataset"] == ds].set_index("model").reindex(MODELS_ORDER)
        
        # Use averagepopularity@K as a proxy for popularity bias
        pop5 = subset["averagepopularity@5"].values
        pop10 = subset["averagepopularity@10"].values
        pop20 = subset["averagepopularity@20"].values
        
        x = np.arange(len(MODELS_ORDER))
        width = 0.25
        
        bars1 = ax.bar(x - width, pop5, width, label="@5", 
                       color=[MODEL_COLORS[m] for m in MODELS_ORDER], alpha=0.9)
        bars2 = ax.bar(x, pop10, width, label="@10", 
                       color=[MODEL_COLORS[m] for m in MODELS_ORDER], alpha=0.65)
        bars3 = ax.bar(x + width, pop20, width, label="@20", 
                       color=[MODEL_COLORS[m] for m in MODELS_ORDER], alpha=0.4)
        
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Popularidad Promedio", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")
    
    # Common legend
    axes[-1].legend(title="Top-K", fontsize=9, loc="upper right")
    
    fig.suptitle("Sesgo de Popularidad por Modelo y Dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_popularity_bias.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 6: Popularity Bias")


# ======================================================================
# Fig 7 — Coverage / Diversity Comparison
# ======================================================================
def fig7_diversity():
    rs = comp[comp["split"] == "rs"].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Coverage@10
    ax = axes[0]
    for i, model in enumerate(MODELS_ORDER):
        model_data = rs[rs["model"] == model]["itemcoverage@10"].values
        pos = [i] * len(model_data)
        ax.scatter(pos, model_data, c=MODEL_COLORS[model], s=100, 
                  marker=MODEL_MARKERS[model], edgecolors="black", linewidth=0.8, zorder=5)
        ax.barh(i, np.mean(model_data), height=0.5, 
               color=MODEL_COLORS[model], alpha=0.3)
        ax.text(np.mean(model_data) + 0.02, i, f"{np.mean(model_data):.3f}", 
               va="center", fontsize=9, fontweight="bold")
    
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels(MODELS_ORDER, fontsize=10)
    ax.set_xlabel("Item Coverage @10", fontsize=11)
    ax.set_title("Cobertura del Catálogo", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    # Gini Index @10 (lower = more diverse)
    ax = axes[1]
    for i, model in enumerate(MODELS_ORDER):
        model_data = rs[rs["model"] == model]["giniindex@10"].values
        pos = [i] * len(model_data)
        ax.scatter(pos, model_data, c=MODEL_COLORS[model], s=100,
                  marker=MODEL_MARKERS[model], edgecolors="black", linewidth=0.8, zorder=5)
        ax.barh(i, np.mean(model_data), height=0.5,
               color=MODEL_COLORS[model], alpha=0.3)
        ax.text(np.mean(model_data) + 0.01, i, f"{np.mean(model_data):.3f}",
               va="center", fontsize=9, fontweight="bold")
    
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels([], fontsize=10)
    ax.set_xlabel("Gini Index @10 (menor = más diverso)", fontsize=11)
    ax.set_title("Índice de Concentración", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    
    fig.suptitle("Diversidad y Cobertura de las Recomendaciones (RS split)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_diversity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 7: Diversity / Coverage")


# ======================================================================
# Fig 8 — Radar Chart Multicriterio
# ======================================================================
def fig8_radar():
    rs = comp[comp["split"] == "rs"].copy()
    rs_costs = costs[costs["split"] == "rs"].copy()
    
    # Average across datasets for each model
    avg_metrics = rs.groupby("model").agg({
        "ndcg@10": "mean",
        "itemcoverage@10": "mean",
        "averagepopularity@10": "mean",
        "giniindex@10": "mean",
    }).reindex(MODELS_ORDER)
    
    avg_costs = rs_costs.groupby("model").agg({
        "avg_train_epoch_sec": "mean",
    }).reindex(MODELS_ORDER)
    
    # Robustness score: average drop at 20%
    rob_data = robust[robust["pct"].isin([20, 100])].copy()
    rob_pivot = rob_data.pivot_table(index=["model", "dataset"], columns="pct", values="ndcg@10")
    if 20 in rob_pivot.columns and 100 in rob_pivot.columns:
        rob_pivot["retention"] = rob_pivot[20] / rob_pivot[100].replace(0, np.nan)
        robustness_score = rob_pivot.groupby("model")["retention"].mean().reindex(MODELS_ORDER).fillna(0)
    else:
        robustness_score = pd.Series([0.5] * len(MODELS_ORDER), index=MODELS_ORDER)
    
    # Normalize all to [0, 1]
    def normalize(s, invert=False):
        s_min, s_max = s.min(), s.max()
        if s_max == s_min:
            return pd.Series([0.5] * len(s), index=s.index)
        norm = (s - s_min) / (s_max - s_min)
        return 1 - norm if invert else norm
    
    dimensions = {
        "Precision": normalize(avg_metrics["ndcg@10"]),
        "Eficiencia": normalize(avg_costs["avg_train_epoch_sec"], invert=True),
        "Robustez": normalize(robustness_score),
        "Diversidad": normalize(avg_metrics["giniindex@10"], invert=True),
        "Novedad": normalize(avg_metrics["averagepopularity@10"], invert=True),
    }
    
    dim_names = list(dimensions.keys())
    N = len(dim_names)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    
    for model in MODELS_ORDER:
        values = [dimensions[dim][model] for dim in dim_names]
        values += values[:1]
        ax.plot(angles, values, color=MODEL_COLORS[model], linewidth=2, label=model)
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, alpha=0.6)
    ax.set_title("Perfil Multicriterio por Paradigma", fontsize=14, 
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True)
    
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_radar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 8: Radar Chart")


# ======================================================================
# Fig 9 — Dataset Statistics
# ======================================================================
def fig9_dataset_stats():
    # Dataset statistics (post 5-core filtering)
    stats = pd.DataFrame({
        "Dataset": ["CiteULike-a", "CiteULike-t", "BibSonomy"],
        "Usuarios": [5536, 3527, 325],
        "Ítems": [15429, 6339, 1607],
        "Interacciones": [200180, 77546, 12683],
        "Densidad": ["0.2344%", "0.3468%", "2.4284%"],
        "Int./Usuario": [36.16, 21.99, 39.02],
        "Int./Ítem": [12.97, 12.23, 7.89],
    })
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    
    # Create table
    table = ax.table(
        cellText=stats.values,
        colLabels=stats.columns,
        cellLoc="center",
        loc="center",
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(stats.columns)):
        cell = table[0, j]
        cell.set_facecolor("#2196F3")
        cell.set_text_props(color="white", fontweight="bold")
    
    # Alternate row colors
    for i in range(1, len(stats) + 1):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(stats.columns)):
            table[i, j].set_facecolor(color)
    
    ax.set_title("Estadisticas de los Datasets (post filtrado 5-core)",
                 fontsize=14, fontweight="bold", pad=20)
    
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_dataset_stats.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Fig 9: Dataset Statistics")


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 8 — GENERATING ALL FIGURES")
    print("=" * 70)
    
    fig1_pipeline()
    fig2_heatmap()
    fig3_pareto()
    fig4_robustness()
    # Fig 5 (CD Diagram) generated by statistical_tests.py
    fig6_popularity_bias()
    fig7_diversity()
    fig8_radar()
    fig9_dataset_stats()
    
    print("\n[OK] Phase 8 complete. All figures saved to:", FIGURES)

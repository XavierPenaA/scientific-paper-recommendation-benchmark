# Phase 5: Popularity Bias Analysis — Key Findings

## 1. Correlation Between Accuracy and Popularity Bias

Across all 36 experiments (6 models × 3 datasets × 2 splits):

| Relationship | Pearson r |
|---|---|
| Recall@10 ↔ AveragePopularity@10 | **+0.1924** |
| Recall@10 ↔ ItemCoverage@10      | **+0.4938** |
| Recall@10 ↔ GiniIndex@10         | **-0.3786** |

> Higher accuracy is **positively correlated** with popularity bias

## 2. Sequential vs. Non-Sequential Models

| Group | Recall@10 | AvgPop@10 | ItemCoverage@10 | GiniIndex@10 |
|---|---|---|---|---|
| Sequential (BERT4Rec, SASRec) | 0.1153 | 19.3100 | 0.4308 | 0.8532 |
| Non-Sequential (ItemKNN, MultiVAE, NeuMF, LightGCN) | 0.3119 | 30.9473 | 0.5514 | 0.7774 |

## 3. Per-Model Diversity Summary

| model    | paradigm          |   recall@10 |   itemcoverage@10 |   averagepopularity@10 |   giniindex@10 |   diversity_score |
|:---------|:------------------|------------:|------------------:|-----------------------:|---------------:|------------------:|
| NeuMF    | Deep CF           |      0.4715 |            0.5291 |                37.3859 |         0.8275 |        -0.337284  |
| MultiVAE | Variational AE    |      0.4544 |            0.6114 |                28.6437 |         0.7528 |        -0.0993704 |
| SASRec   | Sequential (SA)   |      0.1842 |            0.5686 |                11.6648 |         0.7932 |         0.0045162 |
| ItemKNN  | Memory-Based CF   |      0.1719 |            0.7469 |                14.8124 |         0.6179 |         0.299794  |
| LightGCN | Graph-Based       |      0.1496 |            0.3183 |                42.9471 |         0.9115 |        -0.646168  |
| BERT4Rec | Sequential (BERT) |      0.0465 |            0.293  |                26.9551 |         0.9132 |        -0.496261  |

## 4. Pareto-Optimal Configurations

The following model/dataset/split combinations lie on the Pareto frontier
(no other configuration dominates on BOTH recall AND item coverage):

| model    | dataset     | split   |   recall@10 |   itemcoverage@10 |
|:---------|:------------|:--------|------------:|------------------:|
| MultiVAE | citeulike-a | loo     |      0.8143 |            0.7162 |
| MultiVAE | citeulike-a | rs      |      0.7159 |            0.7804 |
| ItemKNN  | citeulike-t | loo     |      0.1849 |            0.8145 |
| ItemKNN  | citeulike-t | rs      |      0.1643 |            0.8297 |

## 5. Extremes

- **Best Recall@10**: MultiVAE on citeulike-a (loo) = **0.8143**
- **Most Diverse**: ItemKNN on citeulike-t (rs) — ItemCoverage@10 = **0.8297**
- **Most Popularity-Biased**: LightGCN on citeulike-t (loo) — AvgPop@10 = **73.1016**
- **Least Popularity-Biased**: SASRec on bibsonomy (rs) — AvgPop@10 = **4.8556**

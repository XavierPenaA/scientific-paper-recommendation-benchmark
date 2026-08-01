# Phase 6: Robustness Analysis

## Recall@10 by Data Fraction

| model    | dataset     |     20 |     50 |    100 |   drop_50% |   drop_20% |
|:---------|:------------|-------:|-------:|-------:|-----------:|-----------:|
| BERT4Rec | bibsonomy   | 0.0074 | 0.0128 | 0.0243 |       47.3 |       69.5 |
| BERT4Rec | citeulike-a | 0.0046 | 0.0206 | 0.0829 |       75.2 |       94.5 |
| BERT4Rec | citeulike-t | 0.0016 | 0.0094 | 0.0173 |       45.7 |       90.8 |
| ItemKNN  | bibsonomy   | 0.024  | 0.0429 | 0.117  |       63.3 |       79.5 |
| ItemKNN  | citeulike-a | 0.0155 | 0.0717 | 0.2163 |       66.9 |       92.8 |
| ItemKNN  | citeulike-t | 0.0102 | 0.0428 | 0.1643 |       74   |       93.8 |
| LightGCN | bibsonomy   | 0.008  | 0.0661 | 0.1501 |       56   |       94.7 |
| LightGCN | citeulike-a | 0.0323 | 0.0355 | 0.1979 |       82.1 |       83.7 |
| LightGCN | citeulike-t | 0.0369 | 0.0629 | 0.0909 |       30.8 |       59.4 |
| MultiVAE | bibsonomy   | 0.006  | 0.0592 | 0.2273 |       74   |       97.4 |
| MultiVAE | citeulike-a | 0.0156 | 0.0726 | 0.7159 |       89.9 |       97.8 |
| MultiVAE | citeulike-t | 0.032  | 0.0882 | 0.2676 |       67   |       88   |
| NeuMF    | bibsonomy   | 0.022  | 0.0247 | 0.3552 |       93   |       93.8 |
| NeuMF    | citeulike-a | 0.012  | 0.0621 | 0.6216 |       90   |       98.1 |
| NeuMF    | citeulike-t | 0.0281 | 0.0634 | 0.5096 |       87.6 |       94.5 |
| SASRec   | bibsonomy   | 0.0223 | 0.0674 | 0.1636 |       58.8 |       86.4 |
| SASRec   | citeulike-a | 0.0165 | 0.0791 | 0.2053 |       61.5 |       92   |
| SASRec   | citeulike-t | 0.0213 | 0.0518 | 0.0825 |       37.2 |       74.2 |


## Key Observations

- **Most robust model** (smallest Recall@10 drop at 20% data): **LightGCN** (avg 79.3% drop)

- **Least robust model** (largest drop at 20% data): **NeuMF** (avg 95.5% drop)

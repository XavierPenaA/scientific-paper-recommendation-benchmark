# Scientific Paper Recommendation Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23505825.svg)](https://doi.org/10.5281/zenodo.23505825)
[![GitHub release](https://img.shields.io/github/v/release/XavierPenaA/scientific-paper-recommendation-benchmark)](https://github.com/XavierPenaA/scientific-paper-recommendation-benchmark/releases/tag/v1.0-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A Multi-Criteria Benchmark of Collaborative Filtering Paradigms for Scientific Paper Recommendation: A Study of Robustness and Generalization**
>
> Xavier Peña · Diego Rivera — Universidad de Cuenca / Universidad Politécnica Salesiana

---

## 📋 Overview

This repository contains all code, configuration files, and reproducibility artifacts for a multi-criteria benchmark evaluating **six collaborative filtering paradigms** for scientific paper recommendation:

| Model | Paradigm |
|---|---|
| **ItemKNN** | Memory-based heuristic |
| **NeuMF** | Neural matrix factorization |
| **MultiVAE** | Variational autoencoder |
| **SASRec** | Transformer (unidirectional) |
| **BERT4Rec** | Transformer (bidirectional) |
| **LightGCN** | Graph neural network |

Evaluated on **3 academic datasets** (CiteULike-a, CiteULike-t, ML-1M adapted) under **2 partition protocols** (Leave-One-Out, Random Split), across **5 evaluation axes**:
- 📊 Offline ranking accuracy (NDCG@10, Recall@10)
- 🌐 Diversity (Catalog Coverage, Gini Index)
- 🧪 Robustness to data scarcity (50% / 20% training)
- ⚡ Computational efficiency (memory, throughput, training time)
- 📐 Statistical rigor (Friedman + Nemenyi post-hoc tests)

---

## 🗂️ Repository Structure

```
.
├── configs/          # RecBole YAML configuration files for each model
├── data/             # Preprocessed datasets (5-core filtered)
├── scripts/          # Preprocessing, training, evaluation, and plot scripts
├── analysis/         # Statistical analysis notebooks and scripts
├── results/          # Raw results CSVs and metric reports
├── paper/            # LaTeX source for the benchmark paper (ES + EN)
│   ├── main.tex      # Spanish version
│   └── main_en.tex   # English version (TechRxiv submission)
├── notebooks/        # Exploratory analysis notebooks
└── requirements.txt  # Python dependencies
```

---

## 🚀 Reproducibility

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train a model

```bash
python scripts/train.py --config configs/itemknn_citeulike_a.yaml
```

### 3. Run full benchmark (all 36 conditions)

```bash
python scripts/run_benchmark.py
```

### 4. Generate figures

```bash
python scripts/generate_figures.py
```

All experiments use **fixed seed 2024** for deterministic reproducibility.

---

## 📦 Archived Release

The exact code state corresponding to the reported results is tagged as [`v1.0-benchmark`](https://github.com/XavierPenaA/scientific-paper-recommendation-benchmark/releases/tag/v1.0-benchmark) and permanently archived at Zenodo:

> **DOI: [10.5281/zenodo.23505825](https://doi.org/10.5281/zenodo.23505825)**

---

## 📄 Citation

If you use this benchmark, please cite:

```bibtex
@misc{pena2026benchmark,
  author       = {Peña, Xavier and Rivera, Diego},
  title        = {A Multi-Criteria Benchmark of Collaborative Filtering Paradigms
                  for Scientific Paper Recommendation},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0-benchmark},
  doi          = {10.5281/zenodo.23505825},
  url          = {https://doi.org/10.5281/zenodo.23505825}
}
```

---

## 👥 Authors

- **Xavier Peña** — Carrera de Computación, Universidad de Cuenca, Ecuador (xavier.pena@ucuenca.edu.ec)
- **Diego Rivera** — Universidad Politécnica Salesiana, Ecuador / IEEE VTS Ecuador Chapter VP

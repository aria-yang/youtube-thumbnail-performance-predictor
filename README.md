# YouTube Thumbnail Performance Predictor

This project predicts relative YouTube thumbnail performance from thumbnail images using a multimodal model built from:

- CNN image embeddings
- OCR-derived text features
- face and face-area features

The target is `normalized_performance = views / subscriber_count`.
For regression, the primary ranking metric is Kendall's Tau. Spearman is retained as a secondary ranking metric for continuity with earlier experiments.

## Repo Layout

- `thumbnail_performance/` core preprocessing, dataset, feature extraction, and inference code
- `training/` training, tuning, ablation, SHAP, and evaluation scripts
- `scripts/` data collection helpers
- `tests/` lightweight preprocessing and contract tests
- `data/` raw data, processed features, and split CSVs
- `models/` saved checkpoints
- `outputs/figures/` report figures
- `outputs/tables/` report tables
- `outputs/metrics/` metric JSON files
- `notebooks/` optional Colab workflows for GPU-backed runs

## Environment

Create the exact environment used for reproduction:

```bash
conda env create -f environment.yml
conda activate youtube-thumbnail-performance-predictor
```
## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         thumbnail_performance and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── thumbnail_performance   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes thumbnail_performance a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


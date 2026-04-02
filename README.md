# YouTube Thumbnail Performance Predictor

This project predicts relative YouTube thumbnail performance using a multimodal model that combines:

- CNN image embeddings
- OCR-derived text features
- face / face-area features
- optional classification and regression training flows

The current final workflow is centered on the merged dataset and the regression model, with GPU-heavy training and fine-tuning run from notebooks.

## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate youtube-thumbnail-performance-predictor
```

## Repo Layout

Main folders:

- `thumbnail_performance/`: reusable package code for preprocessing, features, and inference
- `training/`: training, ablation, SHAP, and cross-split evaluation scripts
- `scripts/`: data collection and baseline helpers
- `notebooks/`: Colab / notebook workflows used for GPU training and fine-tuning
- `data/`: raw, processed, and split artifacts
- `models/`: saved checkpoints
- `outputs/`: generated plots and summary tables

Most important files for reproduction:

- `scripts/fetch_youtube_data.py`
- `thumbnail_performance/dataset.py`
- `thumbnail_performance/ocr_features.py`
- `thumbnail_performance/cnn_embeddings.py`
- `thumbnail_performance/face_emotion_detection.py`
- `training/train_fusion.py`
- `training/train_fusion_regression.py`
- `training/ablation_study.py`
- `training/ablation_study_regression.py`
- `training/run_shap_analysis.py`
- `training/run_shap_regression.py`
- `training/eval_crosssplit.py`
- `training/eval_crosssplit_regression.py`
- `thumbnail_performance/interpretability.py`

## Recreating The Experiment

There are two practical ways to reproduce this project:

1. Use the included processed artifacts and checkpoints already in the repo.
2. Rebuild the preprocessing locally, then run training and fine-tuning from the notebooks on a GPU machine.

The second path is the intended full reproduction workflow.

### 1. Optional: Collect Fresh YouTube Data

If you want to scrape additional metadata and thumbnails, use:

```bash
python scripts/fetch_youtube_data.py --api_key YOUR_API_KEY --channels_file channels.txt --max_per_channel 50
```

This writes:

- `data/raw/new_data.csv`
- `data/thumbnails/new_images/`

If you want to merge that with the existing dataset:

```bash
python scripts/fetch_youtube_data.py --merge
```

That produces:

- `data/raw/merged_data.csv`

If you are only reproducing the final submitted experiment, you can skip this section and use the tracked data already in the repo.

### 2. Build The Labeled CSV

For the original dataset:

```bash
python thumbnail_performance/dataset.py --input-path data/raw/data.csv --output-path data/processed/labeled_data.csv
```

For the merged dataset used by the later experiments:

```bash
python thumbnail_performance/dataset.py --input-path data/raw/merged_data.csv --output-path data/processed/merged_labeled_data.csv
```

This step:

- parses view / subscriber strings
- computes `normalized_performance`
- creates leakage-safe engagement bins for the classification task

### 3. Build OCR Features

For the original dataset:

```bash
python thumbnail_performance/ocr_features.py --csv_path data/processed/labeled_data.csv --thumbnail_dir data/thumbnails/images --ocr_csv_path data/processed/ocr_features.csv --output_path data/processed/text_embeddings.npy
```

For the merged dataset:

```bash
python thumbnail_performance/ocr_features.py --csv_path data/processed/merged_labeled_data.csv --thumbnail_dir data/thumbnails --ocr_csv_path data/processed/merged_ocr_features.csv --output_path data/processed/merged_text_embeddings.npy
```

### 4. Build CNN Embeddings

For the original dataset:

```bash
python thumbnail_performance/cnn_embeddings.py --csv_path data/processed/labeled_data.csv --output_path data/processed/cnn_embeddings.npy
```

The later merged-data training flow can use merged CNN embeddings when available. The training code resolves these in this order:

- `data/processed/merged_cnn_embeddings_resnet50.npy`
- `data/processed/merged_cnn_embeddings.npy`
- `data/processed/cnn_embeddings.npy`

If you are reusing the tracked processed files, you do not need to regenerate them.

### 5. Build Face Features

For the original dataset:

```bash
python thumbnail_performance/face_emotion_detection.py --csv_path data/processed/labeled_data.csv --output_path data/processed/face_embeddings.npy --cache_path data/processed/face_cache.csv
```

For merged-data workflows, later scripts expect merged face artifacts such as:

- `data/processed/merged_face_embeddings.npy`
- `data/processed/merged_face_cache.csv`

If those files already exist in your repo or Drive artifacts, you can reuse them directly.

## GPU Training And Fine-Tuning

The heavyweight training steps were run from notebooks, not from a local CPU-only workflow.

Use these notebooks:

- `notebooks/Training.ipynb`
  Main classification workflow on the original setup
- `notebooks/Discretized.ipynb`
  Updated classification / discretized-bin workflow, ablation, SHAP, and artifact syncing
- `notebooks/Regression.ipynb`
  Final regression workflow, tuning, ablation, SHAP, and cross-split evaluation

Recommended approach:

1. Do local preprocessing first, or rely on the processed artifacts already in the repo.
2. Open the appropriate notebook in Colab or another GPU notebook environment.
3. Mount Drive if you want artifact syncing like the notebook expects.
4. Run the notebook sections in order.

### Final Model Path

For the final submission, the most important notebook is:

- `notebooks/Regression.ipynb`

That notebook covers:

- regression model training
- hyperparameter tuning
- final checkpoint export
- regression ablation study
- regression SHAP analysis
- cross-split generalization evaluation

## Running Training Scripts Directly

You can also run many experiments from Python scripts if you already have the processed artifacts:

Classification:

```bash
python training/train_fusion.py
python training/ablation_study.py
python training/run_shap_analysis.py
python training/eval_crosssplit.py
```

Regression:

```bash
python training/train_fusion_regression.py
python training/tune_fusion_regression.py
python training/ablation_study_regression.py
python training/run_shap_regression.py
python training/eval_crosssplit_regression.py
python training/evaluate_regression_ab.py
```

In practice, we used the notebooks for GPU-backed runs and artifact export.

## Final Outputs

Important saved outputs in the repo include:

- `data/processed/fusion_mlp_regression_metrics.json`
- `data/processed/fusion_regression_tuning_summary.csv`
- `data/processed/fusion_regression_tuning_summary.json`
- `outputs/ablation_table.csv`
- `outputs/shap_feature_importance.csv`
- `outputs/shap_top10_features.csv`
- `outputs/shap_global_importance.png`

Important checkpoints include:

- `models/fusion_mlp.pt`
- `models/fusion_mlp_regression.pt`
- `models/fusion_mlp_regression_final_seed42.pt`

## Interpretability / Demo

To run the final thumbnail scoring and A/B comparison tool:

```bash
python thumbnail_performance/interpretability.py
```

This launches the GUI.

For CLI usage:

```bash
python thumbnail_performance/interpretability.py --cli --image_path PATH_TO_IMAGE --subscriber_count 1000000
```

The interpretability tool supports:

- classification label prediction
- regression-based normalized performance scoring
- A/B thumbnail comparison

## Notes

- The current repo contains both classification and regression workflows.
- The final report should emphasize the regression pipeline, since that is where the later tuning, ablation, SHAP, and cross-split work was added.
- If you are preparing a lean final submission, keep the files needed by the active workflow and remove old scratch files or editor-local config separately.

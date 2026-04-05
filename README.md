# YouTube Thumbnail Performance Predictor

This project predicts relative YouTube thumbnail performance from the thumbnail image itself.
The final workflow uses a multimodal regression model built from:

- CNN image embeddings
- OCR-derived text features
- face and face-area features

The prediction target is `normalized_performance = views / subscriber_count`.

## What Is In This Repo

- `thumbnail_performance/`
  Core preprocessing, feature extraction, dataset logic, and inference code
- `training/`
  Training, ablation, SHAP, and cross-split evaluation scripts
- `scripts/`
  Data collection helpers
- `tests/`
  Small contract and preprocessing tests
- `notebooks/`
  GPU notebook workflows used for the final training runs
- `data/`
  Raw data, processed artifacts, and saved split files
- `models/`
  Saved checkpoints

## Minimal Reproduction

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate youtube-thumbnail-performance-predictor
```

### 2. Use the tracked processed artifacts

The fastest way to reproduce the final project is to use the processed data and checkpoints already in the repo.

### 3. Run the final regression workflow

For GPU-backed training and final experiments, use:

- `notebooks/Training_Regression.ipynb`

This notebook covers:

- regression training
- hyperparameter tuning
- ablation study
- SHAP analysis
- cross-split evaluation

### 4. Run the final demo / inference tool

GUI:

```bash
python thumbnail_performance/interpretability.py
```

CLI:

```bash
python thumbnail_performance/interpretability.py --cli --image_path PATH_TO_IMAGE --subscriber_count 1000000
```

## Rebuilding Features From Scratch

If you want to rebuild the processed artifacts locally instead of using the tracked ones:

```bash
python thumbnail_performance/dataset.py --input-path data/raw/merged_data.csv --output-path data/processed/merged_labeled_data.csv
python thumbnail_performance/ocr_features.py --csv_path data/processed/merged_labeled_data.csv --thumbnail_dir data/thumbnails --ocr_csv_path data/processed/merged_ocr_features.csv --output_path data/processed/merged_text_embeddings.npy
python thumbnail_performance/cnn_embeddings.py --csv_path data/processed/merged_labeled_data.csv --output_path data/processed/merged_cnn_embeddings.npy
python thumbnail_performance/face_emotion_detection.py --csv_path data/processed/merged_labeled_data.csv --output_path data/processed/merged_face_embeddings.npy --cache_path data/processed/merged_face_cache.csv
```

## Main Scripts

- `training/train_fusion.py`
- `training/train_fusion_regression.py`
- `training/tune_fusion_regression.py`
- `training/ablation_study.py`
- `training/ablation_study_regression.py`
- `training/run_shap_analysis.py`
- `training/run_shap_regression.py`
- `training/eval_crosssplit.py`
- `training/eval_crosssplit_regression.py`
- `training/plot_regression_predictions.py`

## Notes

- The final report should emphasize the regression workflow.
- The classification pipeline is still included for comparison and ablation.
- Heavy training was done in notebooks on GPU rather than from a local CPU-only setup.

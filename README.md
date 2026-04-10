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

Optional quick checks:

```bash
python -m pytest tests
python -m py_compile training\train_fusion_regression.py training\tune_fusion_regression.py training\ablation_study_regression.py training\eval_crosssplit_regression.py training\plot_regression_predictions.py
```

## Required Artifacts

Some processed feature files are too large for GitHub and are distributed separately:

- `https://drive.google.com/drive/folders/178JL0JCOksFKrN57nI6LDXIswCRFsHNs?usp=sharing`

Place the downloaded files into the repo as follows:

- `data/processed/`
  - `merged_labeled_data.csv`
  - `merged_text_embeddings.npy`
  - `merged_face_embeddings.npy`
  - `merged_cnn_embeddings_resnet50.npy`
  - `merged_cnn_cache_resnet50.csv`
  - `merged_face_cache.csv`
  - `merged_ocr_features.csv`
  - `fusion_mlp_regression_final_seed42_metrics.json`
- `data/splits/`
  - `random_train.csv`
  - `random_val.csv`
  - `random_test.csv`
  - `channel_train.csv`
  - `channel_val.csv`
  - `channel_test.csv`
  - `time_train.csv`
  - `time_val.csv`
  - `time_test.csv`
- `models/`
  - `fusion_mlp_regression_final_seed42.pt`

## Minimal Reproduction

These commands reproduce the main final-report regression outputs using the final checkpoint and processed artifacts.

### 1. Final regression metrics on the random split

```bash
python training/train_fusion_regression.py --target_transform log1p --loss mse --split_name random --seed 42 --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --metrics_path data/processed/fusion_mlp_regression_final_seed42_metrics.json
```

### 2. Regression ablation study

```bash
python training/ablation_study_regression.py --target_transform log1p --loss mse --split_name random --hidden1 512 --hidden2 256 --dropout 0.4 --ranking_metric test_kendall --output_dir outputs
```

Generated files:

- `outputs/ablation_regression_summary.csv`
- `outputs/ablation_regression_all_runs.csv`
- `outputs/ablation_regression_test_kendall.png`

### 3. Regression SHAP analysis

```bash
python training/run_shap_regression.py --target_transform log1p --loss mse --checkpoint_path models/fusion_mlp_regression_final_seed42.pt
```

### 4. Cross-split generalization evaluation

```bash
python training/eval_crosssplit_regression.py --target_transform log1p --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --output_dir outputs
```

Generated files:

- `outputs/cross_split_regression.csv`
- `outputs/cross_split_regression.json`
- `outputs/cross_split_regression.png`

### 5. Figure 3: predicted vs. true normalized performance

```bash
python training/plot_regression_predictions.py --target_mode log1p --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --display_metrics_from file --display_metrics_file data/processed/fusion_mlp_regression_final_seed42_metrics.json
```

Generated files:

- `outputs/figures/figure3_regression_predictions_vs_ground_truth.png`
- `outputs/tables/figure3_regression_predictions_vs_ground_truth.csv`
- `outputs/metrics/figure3_regression_predictions_vs_ground_truth_metrics.json`

## Classification Baseline

The discretized classification pipeline is kept for comparison and ablation:

```bash
python training/train_fusion.py
python training/ablation_study.py
python training/eval_crosssplit.py
```

## Rebuilding Features From Scratch

If you want to rebuild processed features instead of using the shared artifacts:

```bash
python thumbnail_performance/dataset.py --input-path data/raw/merged_data.csv --output-path data/processed/merged_labeled_data.csv
python thumbnail_performance/ocr_features.py --csv_path data/processed/merged_labeled_data.csv --thumbnail_dir data/thumbnails --ocr_csv_path data/processed/merged_ocr_features.csv --output_path data/processed/merged_text_embeddings.npy
python thumbnail_performance/cnn_embeddings.py --csv_path data/processed/merged_labeled_data.csv --output_path data/processed/merged_cnn_embeddings.npy
python thumbnail_performance/face_emotion_detection.py --csv_path data/processed/merged_labeled_data.csv --output_path data/processed/merged_face_embeddings.npy --cache_path data/processed/merged_face_cache.csv
```

## Optional Colab Workflow

The notebooks are optional and were used for GPU-backed runs:

- `notebooks/Training.ipynb`
- `notebooks/Training_Regression.ipynb`

They support uploading a single artifact ZIP into Colab and unpacking it locally. They are not required for minimal reproduction of the final results listed above.

## Inference Demo

GUI:

```bash
python thumbnail_performance/interpretability.py
```

CLI:

```bash
python thumbnail_performance/interpretability.py --cli --image_path PATH_TO_IMAGE --subscriber_count 1000000
```

## Notes

- The final report should emphasize the regression workflow.
- Kendall's Tau is the main ranking metric for regression, ablation, and cross-split evaluation.
- Spearman is still saved in evaluation outputs as a secondary metric.
- The notebook workflow is optional; the exact commands above are the primary reproduction path.

# YouTube Thumbnail Performance Predictor

This project predicts relative YouTube thumbnail performance from thumbnail images using a multimodal model built from CNN image embeddings, OCR-derived text features, and face features.

The target is `normalized_performance = views / subscriber_count`.
For regression, the primary ranking metric is Kendall's Tau. Spearman is also saved as a secondary ranking metric.

## Repo Layout

- `thumbnail_performance/` preprocessing, dataset, feature extraction, inference
- `training/` training, tuning, ablation, SHAP, evaluation scripts
- `scripts/` data collection helpers
- `tests/` lightweight tests
- `data/` raw data, processed features, split CSVs
- `models/` saved checkpoints
- `outputs/figures/` report figures
- `outputs/tables/` report tables
- `outputs/metrics/` metric JSON files
- `notebooks/` optional Colab workflows for GPU-backed runs

## Environment

Run all commands from the repo root.

```bash
conda env create -f environment.yml
conda activate youtube-thumbnail-performance-predictor
```

Optional checks:

```bash
python -m pytest tests
python -m py_compile training\\train_fusion_regression.py training\\tune_fusion_regression.py training\\ablation_study_regression.py training\\eval_crosssplit_regression.py training\\plot_regression_predictions.py
```

## Required Artifacts

Some processed feature files are too large for GitHub and are distributed separately:

- `https://drive.google.com/drive/folders/178JL0JCOksFKrN57nI6LDXIswCRFsHNs?usp=sharing`

Place these files in the repo:

`data/processed/`
- `merged_labeled_data.csv`
- `merged_text_embeddings.npy`
- `merged_face_embeddings.npy`
- `merged_cnn_embeddings_resnet50.npy`
- `merged_cnn_cache_resnet50.csv`
- `merged_face_cache.csv`
- `merged_ocr_features.csv`
- `fusion_mlp_regression_final_seed42_metrics.json`

`data/splits/`
- `random_train.csv`
- `random_val.csv`
- `random_test.csv`
- `channel_train.csv`
- `channel_val.csv`
- `channel_test.csv`
- `time_train.csv`
- `time_val.csv`
- `time_test.csv`

`models/`
- `fusion_mlp_regression_final_seed42.pt`

## Minimal Reproduction

These commands reproduce the main final-report regression outputs.

### 1. Regression ablation study

```bash
python training/ablation_study_regression.py --target_transform log1p --loss mse --split_name random --hidden1 512 --hidden2 256 --dropout 0.4 --ranking_metric test_kendall --output_dir outputs
```

Writes:
- `outputs/ablation_regression_summary.csv`
- `outputs/ablation_regression_all_runs.csv`
- `outputs/ablation_regression_test_kendall.png`

### 2. Regression SHAP analysis

```bash
python training/run_shap_regression.py --target_transform log1p --loss mse --checkpoint_path models/fusion_mlp_regression_final_seed42.pt
```

Writes:
- `outputs/shap_regression_feature_importance.csv`
- `outputs/shap_regression_top7_features.csv`
- `outputs/shap_regression_global_importance.png`

### 3. Cross-split generalization

```bash
python training/eval_crosssplit_regression.py --target_transform log1p --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --output_dir outputs
```

Writes:
- `outputs/cross_split_regression.csv`
- `outputs/cross_split_regression.json`
- `outputs/cross_split_regression.png`

### 4. Figure 3: predicted vs true normalized performance

```bash
python training/plot_regression_predictions.py --target_mode log1p --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --display_metrics_from file --display_metrics_file data/processed/fusion_mlp_regression_final_seed42_metrics.json
```

Writes:
- `outputs/figures/figure3_regression_predictions_vs_ground_truth.png`
- `outputs/tables/figure3_regression_predictions_vs_ground_truth.csv`
- `outputs/metrics/figure3_regression_predictions_vs_ground_truth_metrics.json`

## Optional: Full Regression Retraining

If you want to retrain the final regression model instead of using the provided checkpoint:

```bash
python training/train_fusion_regression.py --target_transform log1p --loss mse --split_name random --seed 42 --checkpoint_path models/fusion_mlp_regression_final_seed42.pt --metrics_path data/processed/fusion_mlp_regression_final_seed42_metrics.json
```

Optional hyperparameter sweep:

```bash
python training/tune_fusion_regression.py --metric_to_rank val_kendall
```

The sweep is much slower than the evaluation commands above and is not required for minimal reproduction.

### 5. Inference demo

GUI:

```bash
python thumbnail_performance/interpretability.py
```

CLI:

```bash
python thumbnail_performance/interpretability.py --cli --image_path PATH_TO_IMAGE --subscriber_count 1000000
```

## Classification Baseline

The discretized classification pipeline is kept for comparison:

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

Optional GPU-backed notebooks:

- `notebooks/Training.ipynb`
- `notebooks/Training_Regression.ipynb`

They are not required for minimal reproduction. The shell commands above are the primary reproduction path.

## Notes

- The final report should emphasize the regression workflow.
- Kendall's Tau is the main ranking metric for regression, ablation, and cross-split evaluation.
- Spearman is still saved in evaluation outputs as a secondary metric.

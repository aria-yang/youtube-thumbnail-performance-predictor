import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from loguru import logger


# Project imports
from thumbnail_performance.config import DATA_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset, read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion import compute_auroc, load_saved_split_ids, train


DEFAULT_CSV_PATH = PROCESSED_DATA_DIR / "merged_labeled_data.csv"
DEFAULT_TEXT_PATH = PROCESSED_DATA_DIR / "merged_text_embeddings.npy"
DEFAULT_FACE_PATH = PROCESSED_DATA_DIR / "merged_face_embeddings.npy"


def resolve_cnn_path() -> Path:
   """Prefer the explicit ResNet50 artifact name, then fall back to the merged CNN file."""
   candidates = [
       PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
       PROCESSED_DATA_DIR / "merged_cnn_embeddings.npy",
   ]
   for path in candidates:
       if path.exists():
           return path
   return candidates[0]


def set_seed(seed: int):
   """Ensures absolute reproducibility across PyTorch, NumPy, and Python."""
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels_tensor: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
   """Computes inverse frequency class weights to handle imbalance."""
   counts = torch.bincount(labels_tensor, minlength=num_classes).float()
   weights = 1.0 / (counts + 1e-6)
   return weights / weights.sum() * num_classes


def get_real_dataloaders(
   batch_size: int = 64,
   csv_path: Path = DEFAULT_CSV_PATH,
   cnn_path: Path | None = None,
   text_path: Path = DEFAULT_TEXT_PATH,
   face_path: Path = DEFAULT_FACE_PATH,
   split_dir: Path = DATA_DIR / "splits",
   split_name: str = "random",
):
   """Loads the pre-extracted multi-modal features used by the fusion model."""
   logger.info("Loading real ThumbnailDataset features from disk...")
   if cnn_path is None:
       cnn_path = resolve_cnn_path()
  
   dataset = ThumbnailDataset(
       csv_path=csv_path,
       cnn_path=cnn_path,
       text_path=text_path,
       face_path=face_path
   )

   split_df = read_csv_with_fallback(csv_path)
   split_df["Id"] = split_df["Id"].astype(str)
   train_ids, val_ids, _ = load_saved_split_ids(split_dir, split_name)
   id_to_idx = {video_id: idx for idx, video_id in enumerate(split_df["Id"])}
   train_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in train_ids]
   val_indices = [id_to_idx[video_id] for video_id in split_df["Id"] if video_id in val_ids]

   train_ds = Subset(dataset, train_indices)
   val_ds = Subset(dataset, val_indices)

   all_labels = torch.stack([dataset[idx][3] for idx in train_indices])
   class_weights = compute_class_weights(all_labels, num_classes=5)
  
   train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
   val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
  
   return train_loader, val_loader, class_weights


class AblationWrapper(nn.Module):
   """Wraps the FusionMLP to zero-out specific modalities for the ablation study."""
   def __init__(self, base_model, use_text=True, use_face=True):
       super().__init__()
       self.base_model = base_model
       self.use_text = use_text
       self.use_face = use_face


   def forward(self, cnn_feat, text_feat, face_feat):
       if not self.use_text:
           text_feat = torch.zeros_like(text_feat)
       if not self.use_face:
           face_feat = torch.zeros_like(face_feat)
          
       return self.base_model(cnn_feat, text_feat, face_feat)


def run_ablation_experiment(
   batch_size: int = 64,
   csv_path: Path = DEFAULT_CSV_PATH,
   cnn_path: Path | None = None,
   text_path: Path = DEFAULT_TEXT_PATH,
   face_path: Path = DEFAULT_FACE_PATH,
   split_dir: Path = DATA_DIR / "splits",
   split_name: str = "random",
   num_epochs: int = 30,
   lr: float = 1e-3,
   early_stopping_metric: str = "auroc",
   early_stopping_patience: int = 12,
):
   seeds = [42, 43, 44, 45, 46]
   device = "cuda" if torch.cuda.is_available() else "cpu"
  
   # Load Data
   train_loader, val_loader, class_weights = get_real_dataloaders(
       batch_size=batch_size,
       csv_path=csv_path,
       cnn_path=cnn_path,
       text_path=text_path,
       face_path=face_path,
       split_dir=split_dir,
       split_name=split_name,
   )
   class_weights = class_weights.to(device)


   # Dynamically determine feature dimensions from the dataset
   sample_cnn, sample_text, sample_face, _ = next(iter(train_loader))
   ACTUAL_CNN_DIM = sample_cnn.shape[1]
   ACTUAL_TEXT_DIM = sample_text.shape[1]
   ACTUAL_FACE_DIM = sample_face.shape[1]
  
   logger.info(f"Detected Dimensions -> CNN: {ACTUAL_CNN_DIM}, Text: {ACTUAL_TEXT_DIM}, Face: {ACTUAL_FACE_DIM}")


   # Define Configurations
   configs = [
       {"name": "CNN-only", "use_text": False, "use_face": False},
       {"name": "CNN + Text", "use_text": True, "use_face": False},
       {"name": "CNN + Face", "use_text": False, "use_face": True},
       {"name": "CNN + Text + Face", "use_text": True, "use_face": True},
   ]


   results = []
   logger.info(f"Starting Multi-Modal Ablation Study over 5 seeds on {device.upper()}...")


   for config in configs:
       logger.info(f"\n{'='*50}")
       logger.info(f"Evaluating Configuration: {config['name']}")
       logger.info(f"{'='*50}")
      
       for seed in seeds:
           set_seed(seed)
          
           # Use the dimensions detected from the data
           base_model = FusionMLP(
               cnn_dim=ACTUAL_CNN_DIM,
               text_dim=ACTUAL_TEXT_DIM,
               face_dim=ACTUAL_FACE_DIM,
               hidden1=512, hidden2=256, num_classes=5, dropout_p=0.4
           )
          
           model = AblationWrapper(
               base_model,
               use_text=config['use_text'],
               use_face=config['use_face']
           ).to(device)
          
           _ = train(
               model=model, train_loader=train_loader, val_loader=val_loader,
               num_epochs=num_epochs,
               lr=lr,
               device=device,
               class_weights=class_weights,
               early_stopping_metric=early_stopping_metric,
               early_stopping_patience=early_stopping_patience,
           )
          
           auroc = compute_auroc(model, val_loader, num_classes=5, device=device)
          
           results.append({
               "Model": config["name"], "Seed": seed, "AUROC": auroc
           })
           logger.info(f"--> {config['name']} | Seed {seed} | AUROC: {auroc:.4f}")


   return pd.DataFrame(results)


def generate_ablation_outputs(df_results, output_dir="outputs"):
   Path(output_dir).mkdir(exist_ok=True, parents=True)
  
   summary = df_results.groupby("Model")["AUROC"].agg(['mean', 'std']).reset_index()
   summary["AUROC (Mean ± Std)"] = summary.apply(
       lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
   )
  
   summary = summary.sort_values(by="mean").reset_index(drop=True)
   table_path = Path(output_dir) / "ablation_table.csv"
   summary[["Model", "AUROC (Mean ± Std)"]].to_csv(table_path, index=False)
  
   logger.success("\n=== Ablation Results Table ===")
   print(summary[["Model", "AUROC (Mean ± Std)"]].to_string(index=False))
  
   plt.figure(figsize=(10, 6))
   sns.boxplot(
       data=df_results, x="Model", y="AUROC",
       order=summary["Model"], palette="Blues", showfliers=False
   )
   sns.stripplot(
       data=df_results, x="Model", y="AUROC",
       order=summary["Model"], color="black", alpha=0.6, jitter=True
   )
   plt.title("Multi-Modal Ablation Study: Thumbnail Performance Prediction", fontsize=14, pad=15)
   plt.ylabel("Validation AUROC", fontsize=12)
   plt.xlabel("Modality Configuration", fontsize=12)
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   plt.tight_layout()
  
   plot_path = Path(output_dir) / "ablation_plot.png"
   plt.savefig(plot_path, dpi=300)
   logger.success(f"Saved plot to {plot_path}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Run the multimodal ablation study with the merged ResNet50-labeled dataset."
   )
   parser.add_argument(
       "--csv_path",
       type=Path,
       default=DEFAULT_CSV_PATH,
       help="Path to the labeled CSV containing engagement_label.",
   )
   parser.add_argument(
       "--cnn_path",
       type=Path,
       default=None,
       help="Path to CNN embeddings. Defaults to merged ResNet50 embeddings when present.",
   )
   parser.add_argument(
       "--text_path",
       type=Path,
       default=DEFAULT_TEXT_PATH,
       help="Path to text embeddings.",
   )
   parser.add_argument(
       "--face_path",
       type=Path,
       default=DEFAULT_FACE_PATH,
       help="Path to face embeddings.",
   )
   parser.add_argument(
       "--batch_size",
       type=int,
       default=64,
       help="Batch size for the ablation data loaders.",
   )
   parser.add_argument(
       "--num_epochs",
       type=int,
       default=30,
       help="Maximum epochs per ablation run.",
   )
   parser.add_argument(
       "--lr",
       type=float,
       default=1e-3,
       help="Learning rate for ablation runs.",
   )
   parser.add_argument(
       "--early_stopping_metric",
       type=str,
       default="auroc",
       choices=["auroc", "loss", "f1"],
       help="Validation metric to monitor for early stopping.",
   )
   parser.add_argument(
       "--early_stopping_patience",
       type=int,
       default=12,
       help="Epochs to wait for improvement before stopping.",
   )
   parser.add_argument(
       "--split_dir",
       type=Path,
       default=DATA_DIR / "splits",
       help="Directory containing saved split CSV files.",
   )
   parser.add_argument(
       "--split_name",
       type=str,
       default="random",
       choices=["random", "channel", "time"],
       help="Saved split prefix to use.",
   )
   parser.add_argument(
       "--output_dir",
       type=str,
       default="outputs",
       help="Directory where the ablation table and plot will be saved.",
   )
   args = parser.parse_args()

   df = run_ablation_experiment(
       batch_size=args.batch_size,
       csv_path=args.csv_path,
       cnn_path=args.cnn_path,
       text_path=args.text_path,
       face_path=args.face_path,
       split_dir=args.split_dir,
       split_name=args.split_name,
       num_epochs=args.num_epochs,
       lr=args.lr,
       early_stopping_metric=args.early_stopping_metric,
       early_stopping_patience=args.early_stopping_patience,
   )
   generate_ablation_outputs(df, output_dir=args.output_dir)

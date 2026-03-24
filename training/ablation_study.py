import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from loguru import logger


# Project imports
from thumbnail_performance.config import PROCESSED_DATA_DIR
from thumbnail_performance.dataset import ThumbnailDataset
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from training.train_fusion import train, compute_auroc


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


def get_real_dataloaders(batch_size=64):
   """Loads your team's real, pre-extracted multi-modal features."""
   logger.info("Loading real ThumbnailDataset features from disk...")
  
   dataset = ThumbnailDataset(
       csv_path=PROCESSED_DATA_DIR / "labeled_data.csv",
       cnn_path=PROCESSED_DATA_DIR / "cnn_embeddings.npy",
       text_path=PROCESSED_DATA_DIR / "text_embeddings.npy",
       face_path=PROCESSED_DATA_DIR / "face_embeddings.npy"
   )
  
   all_labels = dataset.labels
   class_weights = compute_class_weights(all_labels, num_classes=5)
  
   n_total = len(dataset)
   n_train = int(0.8 * n_total)
   n_val = n_total - n_train
  
   train_ds, val_ds = random_split(dataset, [n_train, n_val])
  
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


def run_ablation_experiment():
   seeds = [42, 43, 44, 45, 46]
   device = "cuda" if torch.cuda.is_available() else "cpu"
  
   # Load Data
   train_loader, val_loader, class_weights = get_real_dataloaders()
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
               num_epochs=30, lr=1e-3, device=device, class_weights=class_weights
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
   df = run_ablation_experiment()
   generate_ablation_outputs(df)
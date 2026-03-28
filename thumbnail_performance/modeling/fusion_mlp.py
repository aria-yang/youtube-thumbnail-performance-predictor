"""
Fusion MLP: Multimodal Classifier
Concatenates CNN embeddings + text features + face features
→ 2 hidden layers → ReLU → Dropout → 5-class logits

Features: early stopping, class weighting, AUROC computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

class FusionMLP(nn.Module):
    """
    Multimodal fusion classifier.

    Input:  [CNN embeddings ‖ text features ‖ face features]
    Output: 5-class logits (raw, pre-softmax)
    """

    def __init__(
        self,
        cnn_dim: int   = 512,   # e.g. ResNet-50 embedding
        text_dim: int  = 768,   # e.g. BERT [CLS] token
        face_dim: int  = 128,   # e.g. ArcFace embedding
        hidden1: int   = 512,
        hidden2: int   = 256,
        num_classes: int = 5,
        dropout_p: float = 0.4,
    ):
        super().__init__()
        input_dim = cnn_dim + text_dim + face_dim   # concatenated size

        self.net = nn.Sequential(
            # ── Hidden layer 1 ──────────────────────────
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # ── Hidden layer 2 ──────────────────────────
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # ── Output head ─────────────────────────────
            nn.Linear(hidden2, num_classes),
            # No softmax here — CrossEntropyLoss handles it internally
        )

    def forward(
        self,
        cnn_feat:  torch.Tensor,   # (B, cnn_dim)
        text_feat: torch.Tensor,   # (B, text_dim)
        face_feat: torch.Tensor,   # (B, face_dim)
    ) -> torch.Tensor:             # (B, num_classes)
        x = torch.cat([cnn_feat, text_feat, face_feat], dim=1)
        return self.net(x)
    
class EarlyStopping:
    """
    Monitors validation loss and stops training when it stops improving.

    Args:
        patience  : epochs to wait after last improvement before stopping
        min_delta : minimum improvement to count as "better"
        verbose   : print messages on improvement / trigger
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, verbose: bool = True):
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None        # saved model weights at best epoch

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if training should stop.
        Saves model state dict whenever val_loss improves.
        """
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"  ✓ Val loss improved {self.best_loss:.4f} → {val_loss:.4f}. Saving model.")
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"  · No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                if self.verbose:
                    print("  ✗ Early stopping triggered.")
                return True
        return False

    def restore_best(self, model: nn.Module):
        """Load the best checkpoint back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print("  ↩ Restored best model weights.")

class EarlyStoppingMax:
    """
    Monitors a validation metric and stops training when it stops improving.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        verbose: bool = True,
        metric_name: str = "metric",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.metric_name = metric_name
        self.best_value = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, metric_value: float, model: nn.Module) -> bool:
        if metric_value > self.best_value + self.min_delta:
            if self.verbose:
                print(
                    f"  âœ“ Val {self.metric_name} improved "
                    f"{self.best_value:.4f} â†’ {metric_value:.4f}. Saving model."
                )
            self.best_value = metric_value
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"  Â· No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                if self.verbose:
                    print("  âœ— Early stopping triggered.")
                return True
        return False

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print("  â†© Restored best model weights.")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    N_SAMPLES    = 2000
    CNN_DIM      = 512
    TEXT_DIM     = 768
    FACE_DIM     = 128
    NUM_CLASSES  = 5
    BATCH_SIZE   = 64
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}\n")

    # ── Synthetic multimodal features ──────────
    cnn_feats  = torch.randn(N_SAMPLES, CNN_DIM)
    text_feats = torch.randn(N_SAMPLES, TEXT_DIM)
    face_feats = torch.randn(N_SAMPLES, FACE_DIM)

    # Imbalanced labels: class 0 dominates to show class weighting effect
    raw_labels = np.random.choice(
        NUM_CLASSES,
        size=N_SAMPLES,
        p=[0.50, 0.20, 0.15, 0.10, 0.05],   # highly skewed
    )
    labels = torch.tensor(raw_labels, dtype=torch.long)

    # ── Class weights ───────────────────────────
    class_weights = compute_class_weights(labels, num_classes=NUM_CLASSES)
    print("Class counts :", torch.bincount(labels).tolist())
    print("Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])

    # ── DataLoaders ─────────────────────────────
    dataset    = TensorDataset(cnn_feats, text_feats, face_feats, labels)
    n_train    = int(0.8 * N_SAMPLES)
    n_val      = N_SAMPLES - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ───────────────────────────────────
    model = FusionMLP(
        cnn_dim=CNN_DIM,
        text_dim=TEXT_DIM,
        face_dim=FACE_DIM,
        hidden1=512,
        hidden2=256,
        num_classes=NUM_CLASSES,
        dropout_p=0.4,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    # ── Train ───────────────────────────────────
    print("\n─── Training ───────────────────────────────")
    history = train(
        model, train_loader, val_loader,
        num_epochs=50,
        lr=1e-3,
        device=DEVICE,
        class_weights=class_weights,
    )

    # ── Final evaluation ────────────────────────
    final_auroc = compute_auroc(model, val_loader, num_classes=NUM_CLASSES, device=DEVICE)
    print(f"\n─── Final Val AUROC (best model): {final_auroc:.4f}")

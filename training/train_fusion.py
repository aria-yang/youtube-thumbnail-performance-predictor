from thumbnail_performance.modeling.fusion_mlp import FusionMLP, EarlyStopping
from thumbnail_performance.dataset import ThumbnailDataset

def train(
    model:       FusionMLP,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:  int   = 50,
    lr:          float = 1e-3,
    device:      str   = "cpu",
    class_weights: torch.Tensor = None,
) -> dict:
    """
    Full training loop with:
      • weighted cross-entropy loss (handles class imbalance)
      • AdamW optimiser
      • early stopping on validation loss
      • per-epoch AUROC reported
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)
    stopper   = EarlyStopping(patience=7, verbose=True)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(1, num_epochs + 1):
        # ── Train ──────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            optimiser.zero_grad()
            logits = model(cnn_feat, text_feat, face_feat)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validate ───────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
                logits   = model(cnn_feat, text_feat, face_feat)
                val_loss += criterion(logits, labels).item() * labels.size(0)
        val_loss /= len(val_loader.dataset)

        val_auroc = compute_auroc(model, val_loader, device=device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f}"
        )

        if stopper.step(val_loss, model):
            break

    stopper.restore_best(model)
    return history

def compute_auroc(
    model:      nn.Module,
    loader:     DataLoader,
    num_classes: int = 5,
    device:     str  = "cpu",
) -> float:
    """
    One-vs-Rest macro-averaged AUROC over a DataLoader.

    Steps
    -----
    1. Collect raw logits from the model (no gradient).
    2. Convert to probabilities via softmax.
    3. Binarise true labels (OvR scheme).
    4. Call sklearn's roc_auc_score with multi_class='ovr', average='macro'.

    Returns
    -------
    float : macro AUROC in [0, 1]; 0.5 = random, 1.0 = perfect.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            cnn_feat, text_feat, face_feat, labels = [t.to(device) for t in batch]
            logits = model(cnn_feat, text_feat, face_feat)           # (B, C)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)  # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)  # (N,)

    # Binarise for OvR comparison
    classes       = list(range(num_classes))
    labels_bin    = label_binarize(all_labels, classes=classes)  # (N, C)

    auroc = roc_auc_score(labels_bin, all_probs, multi_class="ovr", average="macro")
    return float(auroc)


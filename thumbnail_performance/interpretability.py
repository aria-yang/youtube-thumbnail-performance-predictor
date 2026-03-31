from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from torchvision import models
import torch.nn as nn

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is not installed in the active Python environment. "
        "Run this script with the 'youtube-thumbnail-performance-predictor' conda "
        "environment."
    ) from exc

from thumbnail_performance.cnn_embeddings import _embed_image, build_embedding_model
from thumbnail_performance.config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.dataset import read_csv_with_fallback
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from thumbnail_performance.ocr_features import extract_ocr_features
from training.train_fusion import load_saved_split_ids

try:
    from facenet_pytorch import MTCNN
except ModuleNotFoundError:
    MTCNN = None


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_NAMES = {
    0: "Very Low",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}
DEFAULT_CLASSIFICATION_MODEL_CANDIDATES = (
    MODELS_DIR / "fusion_mlp.pt",
)
DEFAULT_REGRESSION_MODEL_CANDIDATES = (
    MODELS_DIR / "fusion_mlp_regression_final_seed42.pt",
    MODELS_DIR / "fusion_mlp_regression.pt",
)
DEFAULT_REGRESSION_TUNING_SUMMARY = PROCESSED_DATA_DIR / "fusion_regression_tuning_summary.json"
DEFAULT_TARGET_COLUMN = "normalized_performance"


def resolve_reference_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_model_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class ExtractedFeatures:
    cnn_features: np.ndarray
    text_features: np.ndarray
    face_features: np.ndarray
    text_metadata: dict[str, Any]
    face_metadata: dict[str, Any]
    cnn_dim: int
    text_dim: int
    face_dim: int


@dataclass
class ThumbnailPrediction:
    image_path: str
    subscriber_count: int
    predicted_class: int
    predicted_label: str
    class_probabilities: dict[str, float]
    predicted_normalized_performance: float
    extracted_features: dict[str, Any]
    note: str


def get_expected_feature_dims(
    cnn_reference_path: Path,
    text_reference_path: Path,
    face_reference_path: Path,
) -> tuple[int, int, int]:
    cnn_dim = int(np.load(cnn_reference_path, mmap_mode="r").shape[1])
    text_dim = int(np.load(text_reference_path, mmap_mode="r").shape[1])
    face_dim = int(np.load(face_reference_path, mmap_mode="r").shape[1])
    return cnn_dim, text_dim, face_dim


def load_checkpoint(model_path: Path, device: str) -> tuple[dict[str, Any] | Any, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return checkpoint, {"state_dict": state_dict, "metadata": metadata}


def load_fusion_model(
    model_path: Path,
    cnn_dim: int,
    text_dim: int,
    face_dim: int,
    num_classes: int,
    device: str,
) -> FusionMLP:
    _, payload = load_checkpoint(model_path, device)
    metadata = payload["metadata"]
    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=metadata.get("hidden1", 512),
        hidden2=metadata.get("hidden2", 256),
        num_classes=num_classes,
        dropout_p=metadata.get("dropout_p", 0.4),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model.to(device)


def _largest_face_area_ratio(boxes: np.ndarray | None, w: int, h: int) -> float:
    if boxes is None or len(boxes) == 0:
        return 0.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    img_area = float(w * h) if w > 0 and h > 0 else 0.0
    return float(np.max(areas)) / img_area if img_area > 0 else 0.0


def build_cnn_model_for_dim(cnn_dim: int, device: str) -> nn.Module:
    if cnn_dim == 2048:
        return build_embedding_model(device=device)
    if cnn_dim == 512:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model.to(device)
    raise ValueError(
        f"Unsupported CNN feature dimension {cnn_dim}. Expected a ResNet-18 "
        "checkpoint (512) or ResNet-50 checkpoint (2048)."
    )


def extract_cnn_features(image_path: Path, device: str, cnn_dim: int) -> np.ndarray:
    embedding_model = build_cnn_model_for_dim(cnn_dim=cnn_dim, device=device)
    embedding = _embed_image(image_path, model=embedding_model, device=device)
    if embedding is None:
        raise RuntimeError(f"Could not extract CNN features from {image_path}")
    return embedding.astype(np.float32)


def extract_text_features(image_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    features = extract_ocr_features(str(image_path))
    vector = np.array(
        [
            features["word_count"],
            features["capital_letter_pct"],
            features["has_numeric"],
            features["char_count"],
        ],
        dtype=np.float32,
    )
    return vector, features


def extract_face_features(image_path: Path, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    if MTCNN is None:
        raise ModuleNotFoundError(
            "facenet-pytorch is not installed in the active Python environment."
        )

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mtcnn = MTCNN(keep_all=True, device=device)

    try:
        boxes, _ = mtcnn.detect(img)
    except Exception:
        boxes = None

    feat_cols = ["num_faces", "largest_face_area_ratio"] + [
        f"emotion_{emotion}" for emotion in EMOTIONS
    ] + ["emotion_unknown"]
    vector = np.zeros(len(feat_cols), dtype=np.float32)
    vector[0] = float(0 if boxes is None else len(boxes))
    vector[1] = float(_largest_face_area_ratio(boxes, w, h))
    vector[-1] = 1.0

    metadata = {
        "num_faces": int(vector[0]),
        "largest_face_area_ratio": float(vector[1]),
        "dominant_emotion": "unknown",
        "status": (
            "Face geometry features were extracted from the uploaded image without "
            "using Keras/DeepFace. Emotion fields were set to unknown."
        ),
    }
    return vector, metadata


def extract_multimodal_features(
    image_path: Path,
    cnn_reference_path: Path,
    text_reference_path: Path,
    face_reference_path: Path,
    device: str,
) -> ExtractedFeatures:
    if not image_path.exists():
        raise FileNotFoundError(f"Thumbnail image not found: {image_path}")

    cnn_dim, text_dim, face_dim = get_expected_feature_dims(
        cnn_reference_path=cnn_reference_path,
        text_reference_path=text_reference_path,
        face_reference_path=face_reference_path,
    )

    cnn_features = extract_cnn_features(image_path=image_path, device=device, cnn_dim=cnn_dim)
    text_features, text_metadata = extract_text_features(image_path=image_path)
    face_features, face_metadata = extract_face_features(image_path=image_path, device=device)

    if len(cnn_features) != cnn_dim:
        raise ValueError(f"Expected CNN dim {cnn_dim}, got {len(cnn_features)}")
    if len(text_features) != text_dim:
        raise ValueError(f"Expected text dim {text_dim}, got {len(text_features)}")
    if len(face_features) != face_dim:
        raise ValueError(f"Expected face dim {face_dim}, got {len(face_features)}")

    return ExtractedFeatures(
        cnn_features=cnn_features,
        text_features=text_features,
        face_features=face_features,
        text_metadata=text_metadata,
        face_metadata=face_metadata,
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
    )


def _to_model_tensors(features: ExtractedFeatures, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cnn_tensor = torch.tensor(features.cnn_features, dtype=torch.float32, device=device).unsqueeze(0)
    text_tensor = torch.tensor(features.text_features, dtype=torch.float32, device=device).unsqueeze(0)
    face_tensor = torch.tensor(features.face_features, dtype=torch.float32, device=device).unsqueeze(0)
    return cnn_tensor, text_tensor, face_tensor


def predict_discretized_bin(
    features: ExtractedFeatures,
    model_path: Path,
    device: str,
) -> tuple[int, str, dict[str, float]]:
    model = load_fusion_model(
        model_path=model_path,
        cnn_dim=features.cnn_dim,
        text_dim=features.text_dim,
        face_dim=features.face_dim,
        num_classes=5,
        device=device,
    )
    cnn_tensor, text_tensor, face_tensor = _to_model_tensors(features, device)

    with torch.no_grad():
        logits = model(cnn_tensor, text_tensor, face_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_class = int(np.argmax(probabilities))
    class_probabilities = {
        f"class_{class_idx}_{CLASS_NAMES[class_idx].lower().replace(' ', '_')}": float(prob)
        for class_idx, prob in enumerate(probabilities)
    }
    return predicted_class, CLASS_NAMES[predicted_class], class_probabilities


def resolve_regression_target_mode(
    explicit_mode: str | None,
    tuning_summary_path: Path,
) -> tuple[str, dict[str, Any]]:
    if explicit_mode:
        return explicit_mode, {}
    if tuning_summary_path.exists():
        summary = json.loads(tuning_summary_path.read_text())
        best_config = summary.get("best_config", {})
        return str(best_config.get("target_mode", "log1p")), summary
    return "log1p", {}


def get_regression_inverse_stats(
    target_mode: str,
    csv_path: Path,
    split_dir: Path,
    split_name: str,
    target_column: str,
) -> tuple[float | None, float | None]:
    normalized_mode = target_mode.lower()
    if normalized_mode not in {"log1p_zscore", "clip_log1p_zscore"}:
        return None, None

    df = read_csv_with_fallback(csv_path).copy()
    df["Id"] = df["Id"].astype(str)
    train_ids, _, _ = load_saved_split_ids(split_dir, split_name)
    train_mask = df["Id"].isin(train_ids).to_numpy()

    if not train_mask.any():
        raise ValueError(f"No training rows found for split '{split_name}' while computing regression stats.")

    arr = df[target_column].astype(float).to_numpy().astype(np.float32)
    if normalized_mode == "clip_log1p_zscore":
        clip_value = float(np.quantile(arr[train_mask], 0.99))
        arr = np.clip(arr, a_min=0.0, a_max=clip_value)

    arr = np.log1p(arr)
    train_values = arr[train_mask]
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


def invert_regression_prediction(
    raw_prediction: float,
    target_mode: str,
    zscore_mean: float | None,
    zscore_std: float | None,
) -> float:
    normalized_mode = target_mode.lower()

    if normalized_mode == "none":
        return raw_prediction
    if normalized_mode in {"log1p", "clip_log1p"}:
        return float(np.expm1(raw_prediction))
    if normalized_mode in {"log1p_zscore", "clip_log1p_zscore"}:
        if zscore_mean is None or zscore_std is None:
            raise ValueError("Regression target mode requires z-score stats, but they were not available.")
        return float(np.expm1(raw_prediction * zscore_std + zscore_mean))

    raise ValueError(
        "Unsupported regression target mode. Choose from: none, log1p, log1p_zscore, "
        "clip_log1p, clip_log1p_zscore."
    )


def predict_regression_views(
    features: ExtractedFeatures,
    model_path: Path,
    target_mode: str,
    zscore_mean: float | None,
    zscore_std: float | None,
    device: str,
) -> float:
    model = load_fusion_model(
        model_path=model_path,
        cnn_dim=features.cnn_dim,
        text_dim=features.text_dim,
        face_dim=features.face_dim,
        num_classes=1,
        device=device,
    )
    cnn_tensor, text_tensor, face_tensor = _to_model_tensors(features, device)

    with torch.no_grad():
        raw_prediction = float(model(cnn_tensor, text_tensor, face_tensor).squeeze().cpu().item())

    predicted_normalized_performance = invert_regression_prediction(
        raw_prediction=raw_prediction,
        target_mode=target_mode,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    )
    return max(predicted_normalized_performance, 0.0)


def predict_thumbnail(
    image_path: Path,
    subscriber_count: int,
    classification_model_path: Path,
    regression_model_path: Path,
    cnn_reference_path: Path,
    text_reference_path: Path,
    face_reference_path: Path,
    regression_target_mode: str,
    regression_zscore_mean: float | None,
    regression_zscore_std: float | None,
    device: str,
) -> ThumbnailPrediction:
    if subscriber_count < 0:
        raise ValueError("subscriber_count must be a non-negative integer.")

    features = extract_multimodal_features(
        image_path=image_path,
        cnn_reference_path=cnn_reference_path,
        text_reference_path=text_reference_path,
        face_reference_path=face_reference_path,
        device=device,
    )
    predicted_class, predicted_label, class_probabilities = predict_discretized_bin(
        features=features,
        model_path=classification_model_path,
        device=device,
    )
    predicted_normalized_performance = predict_regression_views(
        features=features,
        model_path=regression_model_path,
        target_mode=regression_target_mode,
        zscore_mean=regression_zscore_mean,
        zscore_std=regression_zscore_std,
        device=device,
    )

    return ThumbnailPrediction(
        image_path=str(image_path),
        subscriber_count=int(subscriber_count),
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        class_probabilities=class_probabilities,
        predicted_normalized_performance=predicted_normalized_performance,
        extracted_features={
            "ocr": features.text_metadata,
            "face": features.face_metadata,
            "feature_dims": {
                "cnn_dim": features.cnn_dim,
                "text_dim": features.text_dim,
                "face_dim": features.face_dim,
            },
        },
        note=(
            "A/B ranking uses the tuned regression setup to estimate relative normalized "
            "performance. Bin classification uses the discretized FusionMLP."
        ),
    )


def compare_thumbnail_predictions(
    prediction_a: ThumbnailPrediction,
    prediction_b: ThumbnailPrediction,
) -> tuple[str, float, float]:
    score_a = prediction_a.predicted_normalized_performance
    score_b = prediction_b.predicted_normalized_performance

    if np.isclose(score_a, score_b, atol=1e-6):
        return (
            "Both thumbnails are effectively tied by the predicted relative performance score.",
            score_a,
            score_b,
        )

    winner = "Thumbnail A" if score_a > score_b else "Thumbnail B"
    return (
        f"{winner} is expected to perform better.",
        score_a,
        score_b,
    )


def build_prediction_lines(prediction: ThumbnailPrediction) -> list[str]:
    lines = [
        f"Predicted bin: {prediction.predicted_class} ({prediction.predicted_label})",
        f"Predicted relative performance score: {prediction.predicted_normalized_performance:.6f}",
        "",
        "Class probabilities:",
    ]
    for class_idx in range(5):
        class_name = CLASS_NAMES[class_idx]
        key = f"class_{class_idx}_{class_name.lower().replace(' ', '_')}"
        lines.append(f"{class_idx} ({class_name}): {prediction.class_probabilities[key]:.4f}")
    return lines


def print_prediction(prediction: ThumbnailPrediction) -> None:
    print("\nPrediction summary:")
    print(f"  Predicted bin: {prediction.predicted_class} ({prediction.predicted_label})")
    print(f"  Predicted relative performance score: {prediction.predicted_normalized_performance:.6f}")

    print("\nContext:")
    print(f"  Image path: {prediction.image_path}")
    print(f"  Subscriber count: {prediction.subscriber_count}")
    print(f"  OCR text: {prediction.extracted_features['ocr']['raw_text']!r}")
    print(f"  Face summary: {prediction.extracted_features['face']}")
    print(f"  Note: {prediction.note}")


class PillButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        command,
        *,
        width: int = 180,
        height: int = 44,
        radius: int = 22,
        bg_color: str = "#c81d25",
        hover_color: str = "#a8141b",
        text_color: str = "#ffffff",
        font: tuple[str, int, str] = ("Segoe UI", 10, "bold"),
    ) -> None:
        try:
            parent_bg = parent.cget("background")
        except tk.TclError:
            parent_bg = "#111111"
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent_bg,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.command = command
        self.width_px = width
        self.height_px = height
        self.radius = radius
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font_spec = font
        self.text = text
        self._draw(self.bg_color)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _rounded_rect(self, color: str) -> None:
        w = self.width_px
        h = self.height_px
        r = min(self.radius, h // 2, w // 2)
        self.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w - 2 * r, 0, w, 2 * r, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0, h - 2 * r, 2 * r, h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w - r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h - r, fill=color, outline=color)

    def _draw(self, color: str) -> None:
        self.delete("all")
        self._rounded_rect(color)
        self.create_text(
            self.width_px / 2,
            self.height_px / 2,
            text=self.text,
            fill=self.text_color,
            font=self.font_spec,
        )

    def _on_click(self, _event) -> None:
        if self.command is not None:
            self.command()

    def _on_enter(self, _event) -> None:
        self.configure(cursor="hand2")
        self._draw(self.hover_color)

    def _on_leave(self, _event) -> None:
        self._draw(self.bg_color)


class ThumbnailPredictionGUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.image_path_a: Path | None = None
        self.image_path_b: Path | None = None
        self.preview_image_a = None
        self.preview_image_b = None

        self.root = tk.Tk()
        self.root.title("Thumbnail A/B Predictor")
        self.root.geometry("1080x920")
        self.root.configure(bg="#111111")

        self.image_path_var_a = tk.StringVar(value="No image selected for Thumbnail A")
        self.image_path_var_b = tk.StringVar(value="No image selected for Thumbnail B")
        self.subscriber_count_var = tk.StringVar()
        self.verdict_var = tk.StringVar(
            value="Upload two thumbnails to compare which one is expected to perform better and the predicted bins."
        )
        self.abtest_var = tk.StringVar(
            value="A/B testing verdict will appear here."
        )
        self.note_var = tk.StringVar(
            value=(
                "A/B ranking uses the tuned regression setup for relative performance. "
                "Bin output uses the discretized classifier."
            )
        )
        self.detail_var_a = tk.StringVar(value="Thumbnail A details will appear here.")
        self.detail_var_b = tk.StringVar(value="Thumbnail B details will appear here.")

        self._configure_styles()
        self._build_layout()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background="#111111")
        style.configure("Surface.TFrame", background="#181818")
        style.configure("Hero.TFrame", background="#181818")
        style.configure("Card.TLabelframe", background="#181818", borderwidth=1, relief="solid", bordercolor="#343434")
        style.configure("Card.TLabelframe.Label", background="#181818", foreground="#ff4d5a", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", background="#111111", foreground="#ff4d5a", font=("Segoe UI", 24, "bold"))
        style.configure("Body.TLabel", background="#111111", foreground="#d7d7d7", font=("Segoe UI", 11))
        style.configure("CardLabel.TLabel", background="#181818", foreground="#f3f3f3", font=("Segoe UI", 10))
        style.configure("Meta.TLabel", background="#181818", foreground="#ff9aa2", font=("Segoe UI", 9, "bold"))
        style.configure("Section.TLabel", background="#181818", foreground="#ff4d5a", font=("Segoe UI", 12, "bold"))
        style.configure("Modern.TEntry", fieldbackground="#101010", foreground="#f5f5f5", bordercolor="#454545", lightcolor="#ff2f3d", darkcolor="#454545", padding=7)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=18, style="App.TFrame")
        container.pack(fill="both", expand=True)

        hero = ttk.Frame(container, padding=20, style="Hero.TFrame")
        hero.pack(fill="x", pady=(0, 18))

        banner = tk.Frame(hero, bg="#ff2f3d", height=8)
        banner.pack(fill="x", pady=(0, 16))

        ttk.Label(
            hero,
            text="Thumbnail Performance A/B Predictor",
            font=("Segoe UI", 26, "bold"),
            foreground="#ff4d5a",
            background="#181818",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            hero,
            text=(
                "Upload two thumbnail variants, enter subscriber count, see which one is "
                "expected to perform better, and view the discretized bin estimates."
            ),
            wraplength=780,
            foreground="#d7d7d7",
            background="#181818",
        ).pack(anchor="w", pady=(0, 16))

        pill_row = ttk.Frame(hero, style="Hero.TFrame")
        pill_row.pack(fill="x")
        ttk.Label(
            pill_row,
            text="Regression A/B winner",
            style="Meta.TLabel",
        ).pack(side="left", padx=(0, 18))
        ttk.Label(
            pill_row,
            text="Discretized bin output",
            style="Meta.TLabel",
        ).pack(side="left", padx=(0, 18))
        ttk.Label(
            pill_row,
            text="Random split calibration",
            style="Meta.TLabel",
        ).pack(side="left")

        meta_card = ttk.LabelFrame(container, text="Prediction Context", padding=14, style="Card.TLabelframe")
        meta_card.pack(fill="x", pady=(0, 16))

        meta_grid = ttk.Frame(meta_card, style="Surface.TFrame")
        meta_grid.pack(fill="x")
        ttk.Label(meta_grid, text="Subscriber Count:", style="CardLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(meta_grid, textvariable=self.subscriber_count_var, width=28, style="Modern.TEntry").grid(row=0, column=1, sticky="w", padx=(10, 0))

        upload_row = ttk.Frame(container, style="App.TFrame")
        upload_row.pack(fill="both", expand=False, pady=(0, 16))

        card_a = ttk.LabelFrame(upload_row, text="Thumbnail A", padding=14, style="Card.TLabelframe")
        card_a.pack(side="left", fill="both", expand=True, padx=(0, 8))
        card_b = ttk.LabelFrame(upload_row, text="Thumbnail B", padding=14, style="Card.TLabelframe")
        card_b.pack(side="left", fill="both", expand=True, padx=(8, 0))

        PillButton(
            card_a,
            text="Upload Thumbnail A",
            command=lambda: self._select_image("a"),
            width=190,
            height=42,
            radius=21,
            bg_color="#ff2f3d",
            hover_color="#d61f2d",
            text_color="#ffffff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(card_a, textvariable=self.image_path_var_a, wraplength=420, style="CardLabel.TLabel").pack(anchor="w", pady=(10, 10))
        self.preview_label_a = ttk.Label(card_a, text="No image uploaded yet.", style="CardLabel.TLabel", anchor="center")
        self.preview_label_a.pack(fill="x", ipady=18)

        PillButton(
            card_b,
            text="Upload Thumbnail B",
            command=lambda: self._select_image("b"),
            width=190,
            height=42,
            radius=21,
            bg_color="#ff2f3d",
            hover_color="#d61f2d",
            text_color="#ffffff",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(card_b, textvariable=self.image_path_var_b, wraplength=420, style="CardLabel.TLabel").pack(anchor="w", pady=(10, 10))
        self.preview_label_b = ttk.Label(card_b, text="No image uploaded yet.", style="CardLabel.TLabel", anchor="center")
        self.preview_label_b.pack(fill="x", ipady=18)

        action_row = ttk.Frame(container, style="App.TFrame")
        action_row.pack(fill="x", pady=(0, 16))
        PillButton(
            action_row,
            text="Run A/B Prediction",
            command=self._run_prediction,
            width=220,
            height=48,
            radius=24,
            bg_color="#ff2f3d",
            hover_color="#d61f2d",
            text_color="#ffffff",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="center")

        results_frame = ttk.LabelFrame(container, text="Prediction Results", padding=14, style="Card.TLabelframe")
        results_frame.pack(fill="both", expand=True)

        ttk.Label(
            results_frame,
            textvariable=self.verdict_var,
            font=("Helvetica", 15, "bold"),
            wraplength=980,
            style="CardLabel.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            results_frame,
            textvariable=self.abtest_var,
            font=("Helvetica", 12, "bold"),
            wraplength=980,
            style="CardLabel.TLabel",
        ).pack(anchor="w", pady=(0, 12))

        results_grid = ttk.Frame(results_frame, style="Surface.TFrame")
        results_grid.pack(fill="both", expand=True)

        left_results = ttk.LabelFrame(results_grid, text="Thumbnail A Breakdown", padding=12, style="Card.TLabelframe")
        left_results.pack(fill="both", expand=True, pady=(0, 10))
        right_results = ttk.LabelFrame(results_grid, text="Thumbnail B Breakdown", padding=12, style="Card.TLabelframe")
        right_results.pack(fill="both", expand=True)

        self.probability_text_a = tk.Text(
            left_results,
            height=12,
            width=46,
            state="disabled",
            bg="#0f0f0f",
            fg="#f3f3f3",
            insertbackground="#f3f3f3",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#343434",
            font=("Cascadia Code", 10),
            padx=12,
            pady=12,
        )
        self.probability_text_a.pack(fill="x", pady=(0, 12))
        ttk.Label(left_results, textvariable=self.detail_var_a, wraplength=440, style="CardLabel.TLabel").pack(anchor="w")

        self.probability_text_b = tk.Text(
            right_results,
            height=12,
            width=46,
            state="disabled",
            bg="#0f0f0f",
            fg="#f3f3f3",
            insertbackground="#f3f3f3",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#343434",
            font=("Cascadia Code", 10),
            padx=12,
            pady=12,
        )
        self.probability_text_b.pack(fill="x", pady=(0, 12))
        ttk.Label(right_results, textvariable=self.detail_var_b, wraplength=440, style="CardLabel.TLabel").pack(anchor="w")

        ttk.Label(results_frame, textvariable=self.note_var, wraplength=980, style="CardLabel.TLabel").pack(anchor="w", pady=(14, 0))

    def _select_image(self, slot: str) -> None:
        selected = filedialog.askopenfilename(
            title="Select a thumbnail image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        selected_path = Path(selected)
        if slot == "a":
            self.image_path_a = selected_path
            self.image_path_var_a.set(str(selected_path))
        else:
            self.image_path_b = selected_path
            self.image_path_var_b.set(str(selected_path))
        self._update_preview(slot)

    def _update_preview(self, slot: str) -> None:
        image_path = self.image_path_a if slot == "a" else self.image_path_b
        if image_path is None:
            return
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((320, 320))
        preview = ImageTk.PhotoImage(image)
        if slot == "a":
            self.preview_image_a = preview
            self.preview_label_a.configure(image=self.preview_image_a, text="")
        else:
            self.preview_image_b = preview
            self.preview_label_b.configure(image=self.preview_image_b, text="")

    def _parse_subscriber_count(self) -> int:
        raw_value = self.subscriber_count_var.get().strip().replace(",", "")
        if not raw_value:
            raise ValueError("Please enter a subscriber count.")
        return int(raw_value)

    def _run_prediction(self) -> None:
        if self.image_path_a is None or self.image_path_b is None:
            messagebox.showerror(
                "Missing image",
                "Please upload both Thumbnail A and Thumbnail B before running the A/B test.",
            )
            return

        try:
            subscriber_count = self._parse_subscriber_count()
            prediction_a = predict_thumbnail(
                image_path=self.image_path_a,
                subscriber_count=subscriber_count,
                classification_model_path=self.args.classification_model_path,
                regression_model_path=self.args.regression_model_path,
                cnn_reference_path=self.args.cnn_reference_path,
                text_reference_path=self.args.text_reference_path,
                face_reference_path=self.args.face_reference_path,
                regression_target_mode=self.args.regression_target_mode,
                regression_zscore_mean=self.args.regression_zscore_mean,
                regression_zscore_std=self.args.regression_zscore_std,
                device=self.args.device,
            )
            prediction_b = predict_thumbnail(
                image_path=self.image_path_b,
                subscriber_count=subscriber_count,
                classification_model_path=self.args.classification_model_path,
                regression_model_path=self.args.regression_model_path,
                cnn_reference_path=self.args.cnn_reference_path,
                text_reference_path=self.args.text_reference_path,
                face_reference_path=self.args.face_reference_path,
                regression_target_mode=self.args.regression_target_mode,
                regression_zscore_mean=self.args.regression_zscore_mean,
                regression_zscore_std=self.args.regression_zscore_std,
                device=self.args.device,
            )
            abtest_message, score_a, score_b = compare_thumbnail_predictions(prediction_a, prediction_b)
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return

        self.verdict_var.set(
            "Predicted bins and relative scores: "
            f"Thumbnail A = score {prediction_a.predicted_normalized_performance:.6f}, "
            f"bin {prediction_a.predicted_class} ({prediction_a.predicted_label}); "
            f"Thumbnail B = score {prediction_b.predicted_normalized_performance:.6f}, "
            f"bin {prediction_b.predicted_class} ({prediction_b.predicted_label})"
        )
        self.abtest_var.set(
            f"{abtest_message}  Score A = {score_a:.6f}, Score B = {score_b:.6f}"
        )
        self.note_var.set(prediction_a.note)

        detail_a = (
            f"OCR text: {prediction_a.extracted_features['ocr']['raw_text']!r}\n"
            f"Face summary: {prediction_a.extracted_features['face']}"
        )
        detail_b = (
            f"OCR text: {prediction_b.extracted_features['ocr']['raw_text']!r}\n"
            f"Face summary: {prediction_b.extracted_features['face']}"
        )
        self.detail_var_a.set(detail_a)
        self.detail_var_b.set(detail_b)

        probability_lines_a = build_prediction_lines(prediction_a)
        probability_lines_b = build_prediction_lines(prediction_b)

        self.probability_text_a.configure(state="normal")
        self.probability_text_a.delete("1.0", tk.END)
        self.probability_text_a.insert(tk.END, "\n".join(probability_lines_a))
        self.probability_text_a.configure(state="disabled")

        self.probability_text_b.configure(state="normal")
        self.probability_text_b.delete("1.0", tk.END)
        self.probability_text_b.insert(tk.END, "\n".join(probability_lines_b))
        self.probability_text_b.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run combined discretized-bin and regression A/B thumbnail interpretability."
    )
    parser.add_argument(
        "--classification_model_path",
        type=Path,
        default=resolve_model_path(*DEFAULT_CLASSIFICATION_MODEL_CANDIDATES),
    )
    parser.add_argument(
        "--regression_model_path",
        type=Path,
        default=resolve_model_path(*DEFAULT_REGRESSION_MODEL_CANDIDATES),
    )
    parser.add_argument(
        "--cnn_reference_path",
        type=Path,
        default=resolve_reference_path(
            PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
            PROCESSED_DATA_DIR / "merged_cnn_embeddings.npy",
            PROCESSED_DATA_DIR / "cnn_embeddings.npy",
        ),
    )
    parser.add_argument(
        "--text_reference_path",
        type=Path,
        default=resolve_reference_path(
            PROCESSED_DATA_DIR / "merged_text_embeddings.npy",
            PROCESSED_DATA_DIR / "text_embeddings.npy",
        ),
    )
    parser.add_argument(
        "--face_reference_path",
        type=Path,
        default=resolve_reference_path(
            PROCESSED_DATA_DIR / "merged_face_embeddings.npy",
            PROCESSED_DATA_DIR / "face_embeddings.npy",
        ),
    )
    parser.add_argument("--csv_path", type=Path, default=PROCESSED_DATA_DIR / "merged_labeled_data.csv")
    parser.add_argument("--split_dir", type=Path, default=DATA_DIR / "splits")
    parser.add_argument("--regression_split_name", type=str, default="random")
    parser.add_argument("--regression_target_column", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--regression_target_mode", type=str, default=None)
    parser.add_argument("--regression_tuning_summary_path", type=Path, default=DEFAULT_REGRESSION_TUNING_SUMMARY)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cli", action="store_true", help="Use terminal arguments instead of the GUI.")
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--image_path_b", type=Path, default=None)
    parser.add_argument("--subscriber_count", type=int, default=None)
    return parser.parse_args()


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    resolved_mode, summary = resolve_regression_target_mode(
        explicit_mode=args.regression_target_mode,
        tuning_summary_path=args.regression_tuning_summary_path,
    )
    args.regression_target_mode = resolved_mode

    zscore_mean, zscore_std = get_regression_inverse_stats(
        target_mode=args.regression_target_mode,
        csv_path=args.csv_path,
        split_dir=args.split_dir,
        split_name=args.regression_split_name,
        target_column=args.regression_target_column,
    )
    args.regression_zscore_mean = zscore_mean
    args.regression_zscore_std = zscore_std
    return args


def main() -> None:
    args = prepare_args(parse_args())

    if args.cli:
        if args.image_path is None or args.subscriber_count is None:
            raise ValueError("--image_path and --subscriber_count are required when using --cli.")

        prediction = predict_thumbnail(
            image_path=args.image_path,
            subscriber_count=args.subscriber_count,
            classification_model_path=args.classification_model_path,
            regression_model_path=args.regression_model_path,
            cnn_reference_path=args.cnn_reference_path,
            text_reference_path=args.text_reference_path,
            face_reference_path=args.face_reference_path,
            regression_target_mode=args.regression_target_mode,
            regression_zscore_mean=args.regression_zscore_mean,
            regression_zscore_std=args.regression_zscore_std,
            device=args.device,
        )
        print_prediction(prediction)

        if args.image_path_b is not None:
            prediction_b = predict_thumbnail(
                image_path=args.image_path_b,
                subscriber_count=args.subscriber_count,
                classification_model_path=args.classification_model_path,
                regression_model_path=args.regression_model_path,
                cnn_reference_path=args.cnn_reference_path,
                text_reference_path=args.text_reference_path,
                face_reference_path=args.face_reference_path,
                regression_target_mode=args.regression_target_mode,
                regression_zscore_mean=args.regression_zscore_mean,
                regression_zscore_std=args.regression_zscore_std,
                device=args.device,
            )
            print_prediction(prediction_b)
            abtest_message, score_a, score_b = compare_thumbnail_predictions(prediction, prediction_b)
            print("\nA/B test verdict:")
            print(f"  {abtest_message}")
            print(f"  Relative score A: {score_a:.6f}")
            print(f"  Relative score B: {score_b:.6f}")
        return

    app = ThumbnailPredictionGUI(args)
    app.run()


if __name__ == "__main__":
    main()

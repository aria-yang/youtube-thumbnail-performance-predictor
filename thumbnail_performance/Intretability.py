from __future__ import annotations

import argparse
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
from thumbnail_performance.cnn_embeddings import TRANSFORM
from thumbnail_performance.config import MODELS_DIR, PROCESSED_DATA_DIR
from thumbnail_performance.modeling.fusion_mlp import FusionMLP
from thumbnail_performance.ocr_features import extract_ocr_features

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


def resolve_reference_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class ThumbnailPrediction:
    image_path: str
    subscriber_count: int
    predicted_class: int
    predicted_label: str
    class_probabilities: dict[str, float]
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


def load_fusion_model(
    model_path: Path,
    cnn_dim: int,
    text_dim: int,
    face_dim: int,
    device: str,
) -> FusionMLP:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = FusionMLP(
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        hidden1=512,
        hidden2=256,
        num_classes=5,
        dropout_p=0.4,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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


def predict_thumbnail_distribution(
    image_path: Path,
    subscriber_count: int,
    model_path: Path = MODELS_DIR / "fusion_mlp_shap.pt",
    cnn_reference_path: Path = resolve_reference_path(
        PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
        PROCESSED_DATA_DIR / "cnn_embeddings.npy",
    ),
    text_reference_path: Path = resolve_reference_path(
        PROCESSED_DATA_DIR / "merged_text_embeddings.npy",
        PROCESSED_DATA_DIR / "text_embeddings.npy",
    ),
    face_reference_path: Path = resolve_reference_path(
        PROCESSED_DATA_DIR / "merged_face_embeddings.npy",
        PROCESSED_DATA_DIR / "face_embeddings.npy",
    ),
    device: str = "cpu",
) -> ThumbnailPrediction:
    if not image_path.exists():
        raise FileNotFoundError(f"Thumbnail image not found: {image_path}")
    if subscriber_count < 0:
        raise ValueError("subscriber_count must be a non-negative integer.")

    cnn_dim, text_dim, face_dim = get_expected_feature_dims(
        cnn_reference_path=cnn_reference_path,
        text_reference_path=text_reference_path,
        face_reference_path=face_reference_path,
    )

    cnn_features = extract_cnn_features(
        image_path=image_path,
        device=device,
        cnn_dim=cnn_dim,
    )
    text_features, text_metadata = extract_text_features(image_path=image_path)
    face_features, face_metadata = extract_face_features(
        image_path=image_path,
        device=device,
    )

    if len(cnn_features) != cnn_dim:
        raise ValueError(f"Expected CNN dim {cnn_dim}, got {len(cnn_features)}")
    if len(text_features) != text_dim:
        raise ValueError(f"Expected text dim {text_dim}, got {len(text_features)}")
    if len(face_features) != face_dim:
        raise ValueError(f"Expected face dim {face_dim}, got {len(face_features)}")

    model = load_fusion_model(
        model_path=model_path,
        cnn_dim=cnn_dim,
        text_dim=text_dim,
        face_dim=face_dim,
        device=device,
    )

    cnn_tensor = torch.tensor(cnn_features, dtype=torch.float32, device=device).unsqueeze(0)
    text_tensor = torch.tensor(text_features, dtype=torch.float32, device=device).unsqueeze(0)
    face_tensor = torch.tensor(face_features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(cnn_tensor, text_tensor, face_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_class = int(np.argmax(probabilities))
    class_probabilities = {
        f"class_{class_idx}_{CLASS_NAMES[class_idx].lower().replace(' ', '_')}": float(prob)
        for class_idx, prob in enumerate(probabilities)
    }

    return ThumbnailPrediction(
        image_path=str(image_path),
        subscriber_count=int(subscriber_count),
        predicted_class=predicted_class,
        predicted_label=CLASS_NAMES[predicted_class],
        class_probabilities=class_probabilities,
        extracted_features={
            "ocr": text_metadata,
            "face": face_metadata,
            "feature_dims": {
                "cnn_dim": cnn_dim,
                "text_dim": text_dim,
                "face_dim": face_dim,
            },
        },
        note=(
            "Subscriber count is accepted by this interface, but the current FusionMLP "
            "was trained on thumbnail-derived CNN, OCR, and face/emotion features only. "
            "This interface computes those features directly from the uploaded image "
            "before passing them to the FusionMLP."
        ),
    )


def print_prediction(prediction: ThumbnailPrediction) -> None:
    print("\nPrediction distribution:")
    for class_name, probability in prediction.class_probabilities.items():
        print(f"  {class_name}: {probability:.4f}")

    print("\nFinal verdict:")
    print(
        f"  Predicted class = {prediction.predicted_class} "
        f"({prediction.predicted_label})"
    )

    print("\nContext:")
    print(f"  Image path: {prediction.image_path}")
    print(f"  Subscriber count: {prediction.subscriber_count}")
    print(f"  OCR text: {prediction.extracted_features['ocr']['raw_text']!r}")
    print(
        "  Face summary: "
        f"{prediction.extracted_features['face']}"
    )
    print(f"  Note: {prediction.note}")


class ThumbnailPredictionGUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.image_path: Path | None = None
        self.preview_image = None

        self.root = tk.Tk()
        self.root.title("Fusion MLP Thumbnail Predictor")
        self.root.geometry("800x760")

        self.image_path_var = tk.StringVar(value="No image selected")
        self.subscriber_count_var = tk.StringVar()
        self.verdict_var = tk.StringVar(value="Prediction verdict will appear here.")
        self.note_var = tk.StringVar(
            value=(
                "Subscriber count is accepted by the interface, but the current FusionMLP "
                "predicts from CNN, OCR, and face/emotion thumbnail features."
            )
        )
        self.ocr_var = tk.StringVar(value="OCR text: ")
        self.face_var = tk.StringVar(value="Face summary: ")

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Fusion MLP Thumbnail Predictor",
            font=("Helvetica", 18, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            container,
            text="Upload a YouTube thumbnail, enter subscriber count, then generate a prediction.",
            wraplength=740,
        ).pack(anchor="w", pady=(0, 14))

        top_row = ttk.Frame(container)
        top_row.pack(fill="x", pady=(0, 10))

        ttk.Button(
            top_row,
            text="Upload Image",
            command=self._select_image,
        ).pack(side="left")

        ttk.Label(
            top_row,
            textvariable=self.image_path_var,
            wraplength=600,
        ).pack(side="left", padx=(12, 0))

        sub_row = ttk.Frame(container)
        sub_row.pack(fill="x", pady=(0, 14))

        ttk.Label(sub_row, text="Subscriber Count:").pack(side="left")
        ttk.Entry(
            sub_row,
            textvariable=self.subscriber_count_var,
            width=24,
        ).pack(side="left", padx=(12, 0))

        ttk.Button(
            container,
            text="Generate Prediction",
            command=self._run_prediction,
        ).pack(anchor="w", pady=(0, 16))

        preview_frame = ttk.LabelFrame(container, text="Thumbnail Preview", padding=12)
        preview_frame.pack(fill="x", pady=(0, 16))

        self.preview_label = ttk.Label(preview_frame, text="No image uploaded yet.")
        self.preview_label.pack()

        results_frame = ttk.LabelFrame(container, text="Prediction Results", padding=12)
        results_frame.pack(fill="both", expand=True)

        ttk.Label(
            results_frame,
            textvariable=self.verdict_var,
            font=("Helvetica", 14, "bold"),
            wraplength=720,
        ).pack(anchor="w", pady=(0, 12))

        self.probability_text = tk.Text(results_frame, height=8, width=80, state="disabled")
        self.probability_text.pack(fill="x", pady=(0, 12))

        ttk.Label(results_frame, textvariable=self.ocr_var, wraplength=720).pack(anchor="w", pady=(0, 8))
        ttk.Label(results_frame, textvariable=self.face_var, wraplength=720).pack(anchor="w", pady=(0, 8))
        ttk.Label(results_frame, textvariable=self.note_var, wraplength=720).pack(anchor="w")

    def _select_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a thumbnail image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self.image_path = Path(selected)
        self.image_path_var.set(str(self.image_path))
        self._update_preview()

    def _update_preview(self) -> None:
        if self.image_path is None:
            return
        image = Image.open(self.image_path).convert("RGB")
        image.thumbnail((360, 360))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def _parse_subscriber_count(self) -> int:
        raw_value = self.subscriber_count_var.get().strip().replace(",", "")
        if not raw_value:
            raise ValueError("Please enter a subscriber count.")
        return int(raw_value)

    def _run_prediction(self) -> None:
        if self.image_path is None:
            messagebox.showerror("Missing image", "Please upload a thumbnail image first.")
            return

        try:
            subscriber_count = self._parse_subscriber_count()
            prediction = predict_thumbnail_distribution(
                image_path=self.image_path,
                subscriber_count=subscriber_count,
                model_path=self.args.model_path,
                cnn_reference_path=self.args.cnn_reference_path,
                text_reference_path=self.args.text_reference_path,
                face_reference_path=self.args.face_reference_path,
                device=self.args.device,
            )
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return

        self.verdict_var.set(
            f"Predicted class: {prediction.predicted_class} ({prediction.predicted_label})"
        )
        self.ocr_var.set(f"OCR text: {prediction.extracted_features['ocr']['raw_text']!r}")
        self.face_var.set(f"Face summary: {prediction.extracted_features['face']}")
        self.note_var.set(prediction.note)

        probability_lines = [
            f"{label}: {probability:.4f}"
            for label, probability in prediction.class_probabilities.items()
        ]
        self.probability_text.configure(state="normal")
        self.probability_text.delete("1.0", tk.END)
        self.probability_text.insert(tk.END, "\n".join(probability_lines))
        self.probability_text.configure(state="disabled")

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FusionMLP inference for a single thumbnail image."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=MODELS_DIR / "fusion_mlp_shap.pt",
    )
    parser.add_argument(
        "--cnn_reference_path",
        type=Path,
        default=resolve_reference_path(
            PROCESSED_DATA_DIR / "merged_cnn_embeddings_resnet50.npy",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--cli", action="store_true", help="Use terminal arguments instead of the GUI.")
    parser.add_argument("--image_path", type=Path, default=None)
    parser.add_argument("--subscriber_count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cli:
        if args.image_path is None or args.subscriber_count is None:
            raise ValueError("--image_path and --subscriber_count are required when using --cli.")
        prediction = predict_thumbnail_distribution(
            image_path=args.image_path,
            subscriber_count=args.subscriber_count,
            model_path=args.model_path,
            cnn_reference_path=args.cnn_reference_path,
            text_reference_path=args.text_reference_path,
            face_reference_path=args.face_reference_path,
            device=args.device,
        )
        print_prediction(prediction)
        return

    app = ThumbnailPredictionGUI(args)
    app.run()


if __name__ == "__main__":
    main()

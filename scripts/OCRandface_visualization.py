from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Agg")

from facenet_pytorch import MTCNN
from deepface import DeepFace
from PIL import Image

from thumbnail_performance.config import DATA_DIR, FIGURES_DIR
from thumbnail_performance.face_emotion_detection import resolve_thumbnail_path


N_SAMPLES = 4
FACE_BOX_COLOR = "#00FF88"
TEXT_BOX_COLOR = "#FFD700"
CONF_THRESHOLD = 0.4


def get_face_boxes_and_emotions(img: Image.Image, mtcnn: MTCNN) -> list[dict]:
    results = []
    img_np = np.array(img)
    try:
        boxes, probs = mtcnn.detect(img)
    except Exception:
        return results
    if boxes is None:
        return results
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        try:
            face_crop = img_np[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size == 0:
                emotion = "unknown"
            else:
                result = DeepFace.analyze(
                    face_crop, actions=["emotion"], enforce_detection=False, silent=True
                )
                emotion = result[0]["dominant_emotion"]
        except Exception:
            emotion = "unknown"
        results.append({"box": (x1, y1, x2, y2), "emotion": emotion})
    return results


def get_ocr_text(img_path: Path) -> str:
    try:
        import easyocr
        import string
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if w < 640:
            scale = 640 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img_np = np.array(img)
        raw = reader.readtext(img_np)
        tokens = []
        for (_, text, conf) in raw:
            if conf >= CONF_THRESHOLD:
                cleaned = "".join(c for c in text if c in string.printable).strip()
                if cleaned:
                    tokens.append(cleaned)
        return " ".join(tokens) if tokens else "(no text detected)"
    except Exception as e:
        return f"(OCR error: {e})"


def wrap_text(text: str, max_chars: int = 40) -> str:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


def make_qualitative_figure(
    meta_csv: Path,
    thumbnail_dir: Path,
    out_path: Path,
    n_samples: int = N_SAMPLES,
    device: str = "cpu",
    channel_filter: str = None,
) -> None:
    meta = pd.read_csv(meta_csv, index_col=0)
    if channel_filter:
        meta = meta[meta["Channel"] == channel_filter]
    meta = meta.sample(frac=1, random_state=42).reset_index()

    mtcnn = MTCNN(keep_all=True, device=device)
    selected = []
    for _, row in meta.iterrows():
        vid = str(row.name) if "Id" not in meta.columns else str(row["Id"])
        channel = str(row["Channel"])
        img_path = resolve_thumbnail_path(thumbnail_dir, channel, vid)
        if img_path is not None:
            selected.append((vid, channel, img_path, row.get("engagement_label", "?")))
        if len(selected) >= n_samples:
            break

    if not selected:
        print("No thumbnails found on disk — check thumbnail_dir path.")
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 6), facecolor="#1a1a2e")
    if len(selected) == 1:
        axes = [axes]

    fig.suptitle(
        "Figure 1: Examples of Detected Faces, Emotions, and OCR Text",
        fontsize=17, fontweight="bold", color="white", y=1.01
    )

    for ax, (vid, channel, img_path, label) in zip(axes, selected):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        print(f"  Processing {channel} / {vid}...")
        face_results = get_face_boxes_and_emotions(img, mtcnn)
        ocr_text = get_ocr_text(img_path)
        wrapped = wrap_text(ocr_text)

        ax.imshow(img)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

        for face in face_results:
            x1, y1, x2, y2 = face["box"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=FACE_BOX_COLOR, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, max(0, y1 - 5), face["emotion"],
                color=FACE_BOX_COLOR, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
            )

        ax.set_title(
            f"{channel}\nLabel: {label}",
            fontsize=9, color="white", pad=6,
            bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8)
        )
        ax.text(
            w / 2, h + h * 0.04, f"OCR: {wrapped}",
            ha="center", va="top", fontsize=7.5, color=TEXT_BOX_COLOR,
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            transform=ax.transData,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved qualitative figure -> {out_path}")


if __name__ == "__main__":
    make_qualitative_figure(
        meta_csv=DATA_DIR / "splits" / "random_train.csv",
        thumbnail_dir=DATA_DIR / "thumbnails" / "images",
        out_path=FIGURES_DIR / "appendix_qualitative.png",
        n_samples=4,
        device="cpu",
        channel_filter=None
    )

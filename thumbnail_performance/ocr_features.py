"""
Extract OCR-derived thumbnail text features.
"""

import argparse
import os
import re
import string
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

_EASYOCR_READER: Optional[Any] = None


def get_easyocr_reader(languages: Optional[list[str]] = None, gpu: bool = False):
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
    return _EASYOCR_READER


def run_easyocr(img_array: np.ndarray) -> list[tuple[Any, str, float]]:
    reader = get_easyocr_reader()
    return reader.readtext(img_array)


def run_tesseract(img_array: np.ndarray) -> list[tuple[Any, str, float]]:
    import pytesseract

    text = pytesseract.image_to_string(img_array) or ""
    return [([0, 0, 0, 0], text, 1.0)]


def preprocess_image(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < 640:
        scale = 640 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def clean_ocr_text(raw_results: list[tuple[Any, str, float]]) -> str:
    confidence_threshold = 0.4
    tokens = []
    for (_, text, confidence) in raw_results:
        if confidence >= confidence_threshold:
            cleaned = "".join(c for c in text if c in string.printable).strip()
            if cleaned:
                tokens.append(cleaned)
    return " ".join(tokens)


def thumbnail_id_from_path(path_value: str) -> str:
    stem = Path(path_value).stem
    return re.sub(r"[^a-z0-9_-]", "", stem.lower())


def compute_text_features(text: str) -> dict:
    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]

    word_count = len(words)
    capital_letter_pct = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )
    has_numeric = int(bool(re.search(r"\d", text)))
    char_count = len(re.sub(r"\s+", "", text))

    return {
        "word_count": word_count,
        "capital_letter_pct": round(capital_letter_pct, 4),
        "has_numeric": has_numeric,
        "char_count": char_count,
        "raw_text": text,
    }


def extract_ocr_features(img_path: str, backend: str = "easyocr") -> dict:
    try:
        img_array = preprocess_image(img_path)
        if backend == "easyocr":
            raw_results = run_easyocr(img_array)
        elif backend == "tesseract":
            raw_results = run_tesseract(img_array)
        else:
            raise ValueError("backend must be 'easyocr' or 'tesseract'")
        text = clean_ocr_text(raw_results)
    except Exception as exc:
        print(f"  [WARN] OCR failed for {img_path}: {exc}")
        text = ""
    return compute_text_features(text)


def build_ocr_feature_dataframe(
    thumbnail_dir: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
    save_path: Optional[str] = None,
    backend: str = "easyocr",
    valid_ids: Optional[set] = None,
) -> pd.DataFrame:
    image_dir = Path(thumbnail_dir)
    image_paths = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in extensions
            and (valid_ids is None or path.stem in valid_ids)
        ]
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    records = []
    for img_path in tqdm(image_paths, desc="Extracting OCR features"):
        thumbnail_id = img_path.stem
        features = extract_ocr_features(str(img_path), backend=backend)
        features["thumbnail_id"] = thumbnail_id
        records.append(features)

    df = pd.DataFrame(records).set_index("thumbnail_id")
    df = df[
        ["word_count", "capital_letter_pct", "has_numeric", "char_count", "raw_text"]
    ]

    if save_path:
        df.to_csv(save_path)
        print(f"Saved OCR features to {save_path}")

    return df


def demo_single_image(img_path: str, backend: str = "easyocr") -> None:
    features = extract_ocr_features(img_path, backend=backend)
    print(f"\n{'=' * 50}")
    print(f"Thumbnail : {img_path}")
    print(f"Raw text  : {features['raw_text']!r}")
    print(f"Words     : {features['word_count']}")
    print(f"CAPS %    : {features['capital_letter_pct']:.1%}")
    print(f"Numeric   : {'Yes' if features['has_numeric'] else 'No'}")
    print(f"Char count: {features['char_count']}")
    print("=" * 50)


if __name__ == "__main__":
    from thumbnail_performance.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    parser = argparse.ArgumentParser(description="Extract OCR features from thumbnails.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "labeled_data.csv",
        help="Path to labeled CSV used to define row order.",
    )
    parser.add_argument(
        "--thumbnail_dir",
        type=Path,
        default=RAW_DATA_DIR.parent / "thumbnails" / "images",
        help="Root directory containing thumbnails.",
    )
    parser.add_argument(
        "--ocr_csv_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "ocr_features.csv",
        help="Path to save OCR feature CSV.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROCESSED_DATA_DIR / "text_embeddings.npy",
        help="Path to save text feature array.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract"],
        help="OCR engine backend.",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Optional path to a single image for a quick demo.",
    )
    args = parser.parse_args()

    if args.demo:
        demo_single_image(args.demo, backend=args.backend)
    else:
        labeled = pd.read_csv(args.csv_path)
        valid_ids = set(labeled["Id"].astype(str))

        df = build_ocr_feature_dataframe(
            thumbnail_dir=str(args.thumbnail_dir),
            save_path=str(args.ocr_csv_path),
            valid_ids=valid_ids,
            backend=args.backend,
        )

        feat_cols = ["word_count", "capital_letter_pct", "has_numeric", "char_count"]
        arr = (
            df.reindex(labeled["Id"].astype(str))[feat_cols]
            .fillna(0)
            .values.astype(np.float32)
        )
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_path, arr)
        print(f"Saved {args.output_path.name} - shape {arr.shape}")

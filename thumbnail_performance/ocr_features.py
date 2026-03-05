"""
Extract OCR-derived thumbnail text features.

Output features (indexed by thumbnail_id):
  - word_count
  - capital_letter_pct
  - has_numeric
  - char_count
"""

import re
import string
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

_EASYOCR_READER: Optional[Any] = None


def get_easyocr_reader(languages: Optional[list[str]] = None, gpu: bool = False):
    """
    Lazily initialize and reuse EasyOCR reader.
    """
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
    return _EASYOCR_READER


def run_easyocr(img_array: np.ndarray) -> list[tuple[Any, str, float]]:
    reader = get_easyocr_reader()
    return reader.readtext(img_array)


def run_tesseract(img_array: np.ndarray) -> list[tuple[Any, str, float]]:
    """
    Tesseract backend wrapper with EasyOCR-compatible output schema.
    """
    import pytesseract

    text = pytesseract.image_to_string(img_array) or ""
    return [([0, 0, 0, 0], text, 1.0)]


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Light preprocessing to improve OCR accuracy on thumbnails:
      - Upscale small images (OCR degrades below ~300px width)
      - Convert to RGB (handles PNGs with alpha channels)
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < 640:
        scale = 640 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def clean_ocr_text(raw_results: list[tuple[Any, str, float]]) -> str:
    """
    Post-process raw OCR output:
      - Filter low-confidence detections (threshold = 0.4)
      - Strip non-printable characters
      - Collapse extra whitespace
    """
    confidence_threshold = 0.4
    tokens = []
    for (_, text, confidence) in raw_results:
        if confidence >= confidence_threshold:
            cleaned = "".join(c for c in text if c in string.printable).strip()
            if cleaned:
                tokens.append(cleaned)
    return " ".join(tokens)


def thumbnail_id_from_path(path_value: str) -> str:
    """
    Return normalized thumbnail ID from filename/path input.
    """
    stem = Path(path_value).stem
    return re.sub(r"[^a-z0-9_-]", "", stem.lower())


def compute_text_features(text: str) -> dict:
    """
    Compute required text features from a cleaned OCR string.
    """
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
    """
    Extract text features from a single thumbnail image.
    """
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
) -> pd.DataFrame:
    """
    Run OCR extraction over an entire directory of thumbnails.

    Filenames (without extension) are used as thumbnail IDs.
    """
    image_dir = Path(thumbnail_dir)
    image_paths = sorted(
        [path for path in image_dir.iterdir() if path.suffix.lower() in extensions]
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
    """
    Print OCR features for one image.
    """
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract OCR features from YouTube thumbnails."
    )
    parser.add_argument(
        "--thumbnail_dir",
        type=str,
        required=True,
        help="Directory containing thumbnail images.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ocr_features.csv",
        help="Output CSV path for the feature DataFrame.",
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
        help="Path to a single image for a quick demo.",
    )
    args = parser.parse_args()

    if args.demo:
        demo_single_image(args.demo, backend=args.backend)
    else:
        df = build_ocr_feature_dataframe(
            thumbnail_dir=args.thumbnail_dir,
            save_path=args.output_csv,
            backend=args.backend,
        )
        print(f"\nExtracted features for {len(df)} thumbnails.")
        print(df.describe())
        print(df.head())

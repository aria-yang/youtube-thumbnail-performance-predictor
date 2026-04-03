"""
Returns a DataFrame indexed by thumbnail ID with:
  - word_count
  - capital_letter_pct
  - has_numeric (binary)
  - char_count
"""

import os
import re
import string
import warnings
from pathlib import Path

import easyocr
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 1. Initialize EasyOCR reader (GPU if available) ─────────────────────────
# Languages can be extended e.g. ['en', 'fr'] for multilingual thumbnails.
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# ── 2. Preprocessing helpers ─────────────────────────────────────────────────


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Light preprocessing to improve OCR accuracy on thumbnails:
      - Upscale small images (OCR degrades below ~300px width)
      - Convert to RGB (handles PNGs with alpha channels)

    Reference: EasyOCR docs recommend at least 32px text height.
    https://github.com/JaidedAI/EasyOCR#tips-for-better-accuracy
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    # Upscale if width is below 640px to improve character recognition
    if w < 640:
        scale = 640 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def clean_ocr_text(raw_results: list) -> str:
    """
    Post-process raw EasyOCR output:
      - Filter low-confidence detections (threshold = 0.4)
      - Strip non-printable characters
      - Collapse extra whitespace

    Confidence filtering reference:
    https://www.jaided.ai/easyocr/documentation/
    """
    CONFIDENCE_THRESHOLD = 0.4
    tokens = []
    for (_, text, confidence) in raw_results:
        if confidence >= CONFIDENCE_THRESHOLD:
            # Keep only printable ASCII; removes artifacts like box-drawing chars
            cleaned = ''.join(c for c in text if c in string.printable)
            cleaned = cleaned.strip()
            if cleaned:
                tokens.append(cleaned)
    return ' '.join(tokens)


# ── 3. Feature extraction ─────────────────────────────────────────────────────

def extract_ocr_features(img_path: str) -> dict:
    """
    Extract four text features from a single thumbnail image.

    Returns
    -------
    dict with keys:
        word_count          : int   – number of whitespace-separated tokens
        capital_letter_pct  : float – fraction of alpha chars that are uppercase
        has_numeric         : int   – 1 if any digit present, else 0
        char_count          : int   – total non-whitespace character count
        raw_text            : str   – cleaned OCR string (useful for debugging)
    """
    try:
        img_array = preprocess_image(img_path)
        raw_results = reader.readtext(img_array)
        text = clean_ocr_text(raw_results)
    except Exception as e:
        # Gracefully handle corrupt images or read errors
        print(f"  [WARN] OCR failed for {img_path}: {e}")
        text = ""

    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]

    word_count = len(words)
    capital_letter_pct = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars else 0.0
    )
    has_numeric = int(bool(re.search(r'\d', text)))
    char_count = len(text.replace(' ', ''))

    return {
        'word_count': word_count,
        'capital_letter_pct': round(capital_letter_pct, 4),
        'has_numeric': has_numeric,
        'char_count': char_count,
        'raw_text': text,
    }


# ── 4. Batch processing ───────────────────────────────────────────────────────

def build_ocr_feature_dataframe(
    thumbnail_dir: str,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp'),
    save_path: str = None,
) -> pd.DataFrame:
    """
    Run OCR feature extraction over an entire directory of thumbnails.

    Parameters
    ----------
    thumbnail_dir : str
        Path to folder containing thumbnail images.
        Filenames (without extension) are used as thumbnail IDs.
    extensions : tuple
        Image file extensions to include.
    save_path : str, optional
        If provided, saves the DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame indexed by thumbnail_id with OCR feature columns.
    """
    thumbnail_dir = Path(thumbnail_dir)
    image_paths = sorted([
        p for p in thumbnail_dir.iterdir()
        if p.suffix.lower() in extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {thumbnail_dir}")

    records = []
    for img_path in tqdm(image_paths, desc="Extracting OCR features"):
        thumbnail_id = img_path.stem          # e.g. "thumb_0042"
        features = extract_ocr_features(str(img_path))
        features['thumbnail_id'] = thumbnail_id
        records.append(features)

    df = pd.DataFrame(records).set_index('thumbnail_id')

    # Reorder columns for clarity
    df = df[['word_count', 'capital_letter_pct', 'has_numeric',
             'char_count', 'raw_text']]

    if save_path:
        df.to_csv(save_path)
        print(f"Saved OCR features to {save_path}")

    return df


# ── 5. Quick sanity-check on a single image ───────────────────────────────────

def demo_single_image(img_path: str):
    """Print OCR features for one image — useful for debugging."""
    features = extract_ocr_features(img_path)
    print(f"\n{'=' * 50}")
    print(f"Thumbnail : {img_path}")
    print(f"Raw text  : {features['raw_text']!r}")
    print(f"Words     : {features['word_count']}")
    print(f"CAPS %    : {features['capital_letter_pct']:.1%}")
    print(f"Numeric   : {'Yes' if features['has_numeric'] else 'No'}")
    print(f"Char count: {features['char_count']}")
    print('=' * 50)


# ── 6. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract OCR features from YouTube thumbnails."
    )
    parser.add_argument(
        "--thumbnail_dir",
        type=str,
        required=True,
        help="Directory containing thumbnail images."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ocr_features.csv",
        help="Output CSV path for the feature DataFrame."
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Path to a single image for a quick demo."
    )
    args = parser.parse_args()

    if args.demo:
        demo_single_image(args.demo)
    else:
        df = build_ocr_feature_dataframe(
            thumbnail_dir=args.thumbnail_dir,
            save_path=args.output_csv,
        )
        print(f"\nExtracted features for {len(df)} thumbnails.")
        print(df.describe())
        print(df.head())

"""
fetch_youtube_data.py
---------------------
Scrapes YouTube video metadata + thumbnails using the YouTube Data API v3.
Saves NEW data to data/raw/new_data.csv and data/thumbnails/new_images/
so you can train on new data independently or merge with existing.

FOLDER STRUCTURE OUTPUT:
    data/
    ├── raw/
    │   ├── data.csv              <- your existing data (untouched)
    │   ├── new_data.csv          <- freshly scraped (this script)
    │   └── merged_data.csv       <- combined (use --merge flag)
    ├── thumbnails/
    │   ├── images/               <- your existing thumbnails (untouched)
    │   └── new_images/           <- freshly scraped thumbnails (this script)

USAGE:

  Step 1 — Scrape new data:
    python fetch_youtube_data.py \
        --api_key YOUR_KEY \
        --channels_file channels.txt \
        --max_per_channel 50

  Step 2 — (Optional) Merge with existing data:
    python fetch_youtube_data.py --merge

  Step 3 — Run pipeline on NEW data only:
    python thumbnail_performance/dataset.py \
        --input_path data/raw/new_data.csv \
        --output_path data/processed/new_labeled_data.csv

  OR on MERGED data:
    python thumbnail_performance/dataset.py \
        --input_path data/raw/merged_data.csv \
        --output_path data/processed/labeled_data.csv

GET AN API KEY:
    https://console.cloud.google.com/
    -> Enable "YouTube Data API v3"
    -> Credentials -> Create API Key

DAILY QUOTA: 10,000 units/day (free)
    search (resolve handle):      100 units each
    channels (subscriber count):  1 unit each
    playlistItems (video IDs):    1 unit per page (50 videos = 1 unit)
    videos (metadata batch):      1 unit per batch of 50
    With 50 videos/channel: ~103 units/channel -> ~97 channels/day max
    With 10 videos/channel: ~13 units/channel  -> ~760 channels/day max

channels.txt FORMAT (one per line):
    @MrBeast
    @Veritasium
    @3Blue1Brown
"""

import argparse
import time
import requests
import isodate
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths — edit if your project layout differs
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
RAW_DIR         = DATA_DIR / "raw"
THUMB_DIR       = DATA_DIR / "thumbnails"

EXISTING_CSV    = RAW_DIR / "data.csv"
NEW_CSV         = RAW_DIR / "new_data.csv"
MERGED_CSV      = RAW_DIR / "merged_data.csv"

EXISTING_THUMBS = THUMB_DIR / "images"
NEW_THUMBS      = THUMB_DIR / "new_images"

BASE_URL = "https://www.googleapis.com/youtube/v3"

CATEGORY_MAP = {
    "1": "Film", "2": "Cars", "10": "Music", "15": "Pets",
    "17": "Sports", "18": "Travel", "19": "Travel", "20": "Gaming",
    "21": "Vlogging", "22": "People", "23": "Comedy", "24": "Entertainment",
    "25": "News", "26": "Style", "27": "Education", "28": "Science",
    "29": "Activism",
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_channel_id(api_key: str, handle: str) -> str | None:
    """Resolve @handle or channel name to a channel ID. Costs 100 units."""
    handle = handle.strip().split("/")[-1]  # strip full URLs if pasted
    resp = requests.get(f"{BASE_URL}/search", params={
        "part": "snippet", "q": handle,
        "type": "channel", "maxResults": 1, "key": api_key,
    })
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        logger.warning(f"Could not resolve channel: {handle}")
        return None
    return items[0]["snippet"]["channelId"]


def get_channel_info(api_key: str, channel_id: str) -> dict:
    """Returns uploads playlist ID + formatted subscriber string. Costs 1 unit."""
    resp = requests.get(f"{BASE_URL}/channels", params={
        "part": "contentDetails,statistics,snippet",
        "id": channel_id, "key": api_key,
    })
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return {}

    item = items[0]
    n = int(item["statistics"].get("subscriberCount", 0))
    if n >= 1_000_000_000:
        sub_str = f"{n/1e9:.2f}B subscribers"
    elif n >= 1_000_000:
        sub_str = f"{n/1e6:.2f}M subscribers"
    elif n >= 1_000:
        sub_str = f"{n/1e3:.2f}K subscribers"
    else:
        sub_str = f"{n} subscribers"

    return {
        "playlist_id":    item["contentDetails"]["relatedPlaylists"]["uploads"],
        "subscriber_str": sub_str,
        "channel_name":   item["snippet"]["title"],
    }


def get_video_ids(api_key: str, playlist_id: str, max_videos: int, skip: int = 0) -> list[str]:
    """
    Pages through uploads playlist to collect video IDs.
    skip: number of most recent videos to skip before collecting.
    Costs 1 unit/page (50 videos per page).
    """
    video_ids, next_page_token = [], None
    skipped = 0

    while len(video_ids) < max_videos:
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(f"{BASE_URL}/playlistItems", params=params)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            if skipped < skip:
                skipped += 1
                continue
            if len(video_ids) < max_videos:
                video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids


def format_duration(iso_duration: str) -> str:
    """PT13M32S -> '13:32' to match existing CSV format."""
    try:
        total = int(isodate.parse_duration(iso_duration).total_seconds())
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    except Exception:
        return ""


def format_views(n: int) -> str:
    """158000000 -> '158.0M views' to match existing CSV format."""
    if n >= 1_000_000_000:
        return f"{n/1e9:.1f}B views"
    elif n >= 1_000_000:
        return f"{n/1e6:.1f}M views"
    elif n >= 1_000:
        return f"{n/1e3:.1f}K views"
    return f"{n} views"


def get_video_details(api_key: str, video_ids: list[str], subscriber_str: str) -> list[dict]:
    """Fetches metadata for up to 50 videos. Costs 1 unit per batch."""
    resp = requests.get(f"{BASE_URL}/videos", params={
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids), "key": api_key,
    })
    resp.raise_for_status()

    rows = []
    for item in resp.json().get("items", []):
        snippet = item.get("snippet", {})
        stats   = item.get("statistics", {})
        content = item.get("contentDetails", {})
        thumbs  = snippet.get("thumbnails", {})

        thumb_url = (
            thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}
        ).get("url", "")

        rows.append({
            "video_id":       item["id"],
            "channel_name":   snippet.get("channelTitle", ""),
            "subscriber_str": subscriber_str,
            "title":          snippet.get("title", ""),
            "published_at":   snippet.get("publishedAt", "")[:10],
            "view_count":     int(stats.get("viewCount", 0)),
            "views_str":      format_views(int(stats.get("viewCount", 0))),
            "category":       CATEGORY_MAP.get(snippet.get("categoryId", ""), "Entertainment"),
            "duration":       format_duration(content.get("duration", "")),
            "thumbnail_url":  thumb_url,
        })
    return rows


def download_thumbnail(video_id: str, url: str, out_dir: Path) -> bool:
    out_path = out_dir / f"{video_id}.jpg"
    if out_path.exists():
        return True
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        logger.warning(f"Thumbnail failed for {video_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Core scraping function
# ---------------------------------------------------------------------------

def scrape(api_key: str, channel_handles: list[str], max_per_channel: int, skip: int = 0):
    """
    Scrapes all channels. Saves to:
      data/raw/new_data.csv
      data/thumbnails/new_images/
    Existing data is never touched.
    skip: number of most recent videos to skip per channel (use 50 on day 2, 100 on day 3, etc.)
    """
    NEW_THUMBS.mkdir(parents=True, exist_ok=True)
    NEW_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Build set of already-collected IDs to avoid duplicates across both datasets
    existing_ids = set()
    for csv in [EXISTING_CSV, NEW_CSV]:
        if csv.exists():
            existing_ids.update(pd.read_csv(csv)["Id"].astype(str).tolist())
    logger.info(f"Skipping {len(existing_ids)} already-collected video IDs")

    all_rows = []

    for handle in tqdm(channel_handles, desc="Channels"):
        logger.info(f"\n── Scraping: {handle}")

        channel_id = get_channel_id(api_key, handle)
        if not channel_id:
            continue

        info = get_channel_info(api_key, channel_id)
        if not info:
            continue

        video_ids = get_video_ids(api_key, info["playlist_id"], max_per_channel, skip=skip)
        new_ids   = [v for v in video_ids if v not in existing_ids]

        if not new_ids:
            logger.info("  No new videos, skipping.")
            continue

        logger.info(f"  {len(new_ids)} new videos found")

        for i in range(0, len(new_ids), 50):
            batch   = new_ids[i:i+50]
            details = get_video_details(api_key, batch, info["subscriber_str"])

            for d in details:
                if not download_thumbnail(d["video_id"], d["thumbnail_url"], NEW_THUMBS):
                    continue  # Skip rows with no thumbnail (pipeline needs the image)

                # Columns match data.csv exactly
                all_rows.append({
                    "Id":          d["video_id"],
                    "Channel":     d["channel_name"],
                    "Subscribers": d["subscriber_str"],
                    "Title":       d["title"],
                    "URL":         f"https://www.youtube.com/watch?v={d['video_id']}",
                    "Released":    d["published_at"],
                    "Views":       d["views_str"],
                    "Category":    d["category"],
                    "Length":      d["duration"],
                })
                existing_ids.add(d["video_id"])

            time.sleep(0.1)  # stay well within rate limits

        logger.success(f"  Done: {len(new_ids)} videos from {handle}")

    if not all_rows:
        logger.warning("No new data collected. Check your API key and channels.txt.")
        return

    new_df = pd.DataFrame(all_rows)
    new_df.to_csv(NEW_CSV, index=False)

    logger.success(f"\n{'='*50}")
    logger.success(f"Saved {len(new_df)} new rows  ->  {NEW_CSV}")
    logger.success(f"Thumbnails saved             ->  {NEW_THUMBS}/")
    logger.info("\nNext steps:")
    logger.info("  Train on NEW data only:")
    logger.info(f"    python thumbnail_performance/dataset.py --input_path {NEW_CSV} --output_path data/processed/new_labeled_data.csv")
    logger.info("  Or merge first, then run pipeline on merged:")
    logger.info("    python fetch_youtube_data.py --merge")
    logger.info(f"    python thumbnail_performance/dataset.py --input_path {MERGED_CSV} --output_path data/processed/labeled_data.csv")


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge():
    """Combines data.csv + new_data.csv -> merged_data.csv, deduped by Id."""
    if not EXISTING_CSV.exists():
        logger.error(f"Missing {EXISTING_CSV}")
        return
    if not NEW_CSV.exists():
        logger.error(f"Missing {NEW_CSV} — run scrape step first.")
        return

    old_df  = pd.read_csv(EXISTING_CSV)
    new_df  = pd.read_csv(NEW_CSV)
    merged  = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["Id"])
    merged.to_csv(MERGED_CSV, index=False)

    logger.success(f"Merged: {len(old_df)} existing + {len(new_df)} new = {len(merged)} total")
    logger.success(f"Saved -> {MERGED_CSV}")
    logger.info("\nRun the full pipeline on merged data:")
    logger.info(f"  python thumbnail_performance/dataset.py --input_path {MERGED_CSV} --output_path data/processed/labeled_data.csv")
    logger.info(f"  python datasplitting")
    logger.info(f"  python thumbnail_performance/cnn_embeddings.py")
    logger.info(f"  python thumbnail_performance/face_emotion_detection.py")
    logger.info(f"  python thumbnail_performance/ocr_features.py --thumbnail_dir data/thumbnails/images")
    logger.info(f"  python training/train_fusion.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube thumbnail scraper")
    parser.add_argument("--api_key",         type=str, default=None,
                        help="YouTube Data API v3 key (required unless --merge)")
    parser.add_argument("--channels_file",   type=str, default="channels.txt",
                        help="Path to text file with one channel handle per line")
    parser.add_argument("--max_per_channel", type=int, default=50,
                        help="Max videos to pull per channel (default: 50)")
    parser.add_argument("--skip",            type=int, default=0,
                        help="Number of most recent videos to skip per channel. Use 50 on day 2, 100 on day 3, etc.")
    parser.add_argument("--merge",           action="store_true",
                        help="Skip scraping, just merge existing + new CSVs")
    args = parser.parse_args()

    if args.merge:
        merge()
    else:
        if not args.api_key:
            raise ValueError("--api_key is required. Get one at console.cloud.google.com -> YouTube Data API v3")
        channels_path = Path(args.channels_file)
        if not channels_path.exists():
            raise FileNotFoundError(f"Could not find {channels_path}")
        handles = [l.strip() for l in channels_path.read_text().splitlines() if l.strip()]
        logger.info(f"Loaded {len(handles)} channels from {channels_path}")
        scrape(args.api_key, handles, args.max_per_channel, skip=args.skip)
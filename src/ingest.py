"""Ingest ABO catalog: download the public listings JSONL shards directly from
the ABO S3 bucket, filter by category, look up image paths from the public
images metadata CSV, download small images, and write a clean catalog.csv.

Usage:
    python -m src.ingest --category furniture --limit 4000
    python -m src.ingest --category furniture --limit 200   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_IMG_DIR = RAW_DIR / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
CATALOG_CSV = PROCESSED_DIR / "catalog.csv"
IMAGE_META_PATH = RAW_DIR / "abo_images_meta.csv"

# Public ABO bucket
ABO_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
ABO_IMG_META_URL = f"{ABO_BASE}/images/metadata/images.csv.gz"
ABO_LISTINGS_SHARDS = 16  # listings_0.json.gz .. listings_15.json.gz


def _first_lang_value(field: Any) -> str | None:
    if field is None:
        return None
    if isinstance(field, str):
        return field
    if isinstance(field, list) and field:
        for item in field:
            if isinstance(item, dict):
                tag = str(item.get("language_tag") or "")
                if tag.startswith("en"):
                    return item.get("value")
        for item in field:
            if isinstance(item, dict) and "value" in item:
                return item["value"]
    if isinstance(field, dict):
        return field.get("value")
    return None


def _row_category(row: dict) -> str | None:
    cat = row.get("product_type")
    if isinstance(cat, list) and cat:
        cat = cat[0]
    if isinstance(cat, dict):
        cat = cat.get("value")
    return str(cat).lower() if cat else None


def _row_title(row: dict) -> str | None:
    return _first_lang_value(row.get("item_name"))


def _row_description(row: dict) -> str | None:
    desc = row.get("product_description")
    if desc:
        v = _first_lang_value(desc)
        if v:
            return v
    bullets = row.get("bullet_point")
    if isinstance(bullets, list) and bullets:
        parts = []
        for b in bullets:
            if isinstance(b, dict):
                tag = str(b.get("language_tag") or "")
                if tag.startswith("en"):
                    parts.append(b.get("value", ""))
        if parts:
            return " ".join(parts)
    return None


def _row_color(row: dict) -> str | None:
    return _first_lang_value(row.get("color"))


def _load_image_meta(session: requests.Session) -> dict[str, str]:
    """image_id -> S3 path (e.g. '49/49eUv...jpg'). Cached locally."""
    if not IMAGE_META_PATH.exists():
        print(f"[ingest] downloading image metadata from {ABO_IMG_META_URL}")
        r = session.get(ABO_IMG_META_URL, timeout=120)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            IMAGE_META_PATH.write_text(f.read())
    mapping: dict[str, str] = {}
    with IMAGE_META_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["image_id"]] = row["path"]
    print(f"[ingest] image metadata: {len(mapping)} entries")
    return mapping


def _download_image(image_id: str, image_path: str, dest: Path,
                    session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = f"{ABO_BASE}/images/small/{image_path}"
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return False
        Image.open(io.BytesIO(r.content)).verify()
        dest.write_bytes(r.content)
        return True
    except Exception:
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False


def _stream_listings(session: requests.Session) -> Iterable[dict]:
    """Stream rows from all listings_*.json.gz shards in the public S3 bucket.
    Skips shards that 404 (the actual shard count is not documented; we just try
    a generous range and stop on consecutive misses)."""
    misses = 0
    for i in range(ABO_LISTINGS_SHARDS):
        url = f"{ABO_BASE}/listings/metadata/listings_{i}.json.gz"
        print(f"[ingest] fetching {url}", flush=True)
        r = session.get(url, timeout=120)
        if r.status_code == 404:
            misses += 1
            print(f"[ingest] {url} -> 404, skipping", flush=True)
            if misses >= 2:
                print("[ingest] two consecutive 404s, stopping shard scan", flush=True)
                return
            continue
        misses = 0
        r.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            for line in gz:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="furniture")
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--scan-cap", type=int, default=200_000)
    args = parser.parse_args()

    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    image_paths = _load_image_meta(session)

    rows: list[dict] = []
    pbar = tqdm(total=args.limit, desc="kept")
    scanned = 0

    cat_lower = args.category.lower()
    # Furniture in ABO is encoded as product_type values like 'CHAIR', 'SOFA',
    # 'TABLE', 'BED'... rather than literally 'furniture'. We accept either an
    # exact substring match or a built-in furniture set.
    FURNITURE_TYPES = {
        "chair", "sofa", "table", "bed", "ottoman", "bookcase", "desk",
        "shelf", "stool", "bench", "cabinet", "dresser", "nightstand",
        "wardrobe", "futon", "loveseat", "recliner", "armchair", "headboard",
        "console_table", "dining_table", "coffee_table", "side_table",
        "end_table", "tv_stand", "shelving_unit", "bunk_bed",
    }

    for row in _stream_listings(session):
        scanned += 1
        if scanned > args.scan_cap or len(rows) >= args.limit:
            break
        cat = _row_category(row)
        if cat is None:
            continue
        if cat_lower == "furniture":
            if cat not in FURNITURE_TYPES:
                continue
        elif cat_lower not in cat:
            continue
        title = _row_title(row)
        image_id = row.get("main_image_id")
        desc = _row_description(row)
        if not (title and image_id and desc):
            continue
        path = image_paths.get(image_id)
        if not path:
            continue
        dest = RAW_IMG_DIR / f"{image_id}.jpg"
        if not _download_image(image_id, path, dest, session):
            continue
        rows.append({
            "product_id": row.get("item_id") or image_id,
            "title": title,
            "description": desc,
            "category": cat,
            "color": _row_color(row),
            "image_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        })
        pbar.update(1)
    pbar.close()
    print(f"[ingest] scanned {scanned} rows, kept {len(rows)}")

    if not rows:
        print("[ingest] ERROR: no rows kept.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)[
        ["product_id", "title", "description", "category", "color", "image_path"]
    ]
    df.to_csv(CATALOG_CSV, index=False)
    print(f"[ingest] wrote {len(df)} rows -> {CATALOG_CSV}")


if __name__ == "__main__":
    main()

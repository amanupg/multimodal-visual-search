"""CLIP + FAISS retrieval: text or image query -> top-K products."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "processed" / "catalog.csv"
EMB_DIR = ROOT / "data" / "embeddings"
INDEX_PATH = EMB_DIR / "faiss.index"
ID_MAP_PATH = EMB_DIR / "id_map.json"

MODEL_NAME = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(_device()).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


@lru_cache(maxsize=1)
def _index():
    return faiss.read_index(str(INDEX_PATH))


@lru_cache(maxsize=1)
def _id_map() -> dict[int, str]:
    raw = json.loads(ID_MAP_PATH.read_text())
    return {int(k): v for k, v in raw.items()}


@lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_CSV)
    df["product_id"] = df["product_id"].astype(str)
    return df.set_index("product_id", drop=False)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _to_tensor(out) -> "torch.Tensor":
    if torch.is_tensor(out):
        return out
    return getattr(out, "text_embeds", None) or getattr(out, "image_embeds", None) or out.pooler_output


def encode_text(query: str) -> np.ndarray:
    model, processor = _model()
    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(_device())
        feats = _to_tensor(model.get_text_features(**inputs)).cpu().numpy().astype(np.float32)
    return _normalize(feats)


def encode_image(img: Image.Image) -> np.ndarray:
    model, processor = _model()
    with torch.no_grad():
        inputs = processor(images=[img.convert("RGB")], return_tensors="pt").to(_device())
        feats = _to_tensor(model.get_image_features(**inputs)).cpu().numpy().astype(np.float32)
    return _normalize(feats)


def _search(vec: np.ndarray, k: int) -> list[dict]:
    scores, idxs = _index().search(vec, k)
    catalog = _catalog()
    id_map = _id_map()
    out: list[dict] = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0:
            continue
        pid = id_map.get(idx)
        if pid is None or pid not in catalog.index:
            continue
        row = catalog.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        out.append({
            "product_id": pid,
            "title": row["title"],
            "description": row.get("description", ""),
            "category": row.get("category", ""),
            "color": row.get("color", ""),
            "image_path": row["image_path"],
            "score": float(score),
        })
    return out


def search_text(query: str, k: int = 5) -> list[dict]:
    return _search(encode_text(query), k)


def search_image(img: Image.Image, k: int = 5) -> list[dict]:
    return _search(encode_image(img), k)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    for r in search_text(args.query, args.k):
        print(f"{r['score']:.3f}  {r['product_id']}  {r['title'][:80]}")

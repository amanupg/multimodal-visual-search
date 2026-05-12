"""Encode catalog images with CLIP -> embeddings.npy + id_map.json.

Usage:
    python -m src.embed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "processed" / "catalog.csv"
EMB_DIR = ROOT / "data" / "embeddings"
EMB_NPY = EMB_DIR / "embeddings.npy"
ID_MAP = EMB_DIR / "id_map.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH = 32


def _load_model(device: str):
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CATALOG_CSV)
    print(f"[embed] {len(df)} products from {CATALOG_CSV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embed] device={device}")
    model, processor = _load_model(device)

    embeddings: list[np.ndarray] = []
    id_map: dict[int, str] = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH), desc="embed"):
            batch = df.iloc[start : start + BATCH]
            images: list[Image.Image] = []
            kept_idx: list[int] = []
            for offset, (_, row) in enumerate(batch.iterrows()):
                p = ROOT / row["image_path"]
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                images.append(img)
                kept_idx.append(start + offset)
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            if not torch.is_tensor(feats):
                feats = getattr(feats, "image_embeds", None) or feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy().astype(np.float32))
            for global_idx in kept_idx:
                id_map[len(id_map)] = str(df.iloc[global_idx]["product_id"])

    if not embeddings:
        raise RuntimeError("No embeddings produced; check image_path entries.")

    arr = np.concatenate(embeddings, axis=0)
    np.save(EMB_NPY, arr)
    ID_MAP.write_text(json.dumps(id_map))
    print(f"[embed] wrote {arr.shape} -> {EMB_NPY}")
    print(f"[embed] id_map ({len(id_map)} entries) -> {ID_MAP}")


if __name__ == "__main__":
    main()

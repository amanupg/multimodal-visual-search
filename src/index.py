"""Build a FAISS IndexFlatIP over normalized CLIP embeddings.

Usage:
    python -m src.index
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"
EMB_NPY = EMB_DIR / "embeddings.npy"
INDEX_PATH = EMB_DIR / "faiss.index"


def main() -> None:
    arr = np.load(EMB_NPY).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[index] {arr.shape[0]} vectors, dim={arr.shape[1]} -> {INDEX_PATH}")


if __name__ == "__main__":
    main()

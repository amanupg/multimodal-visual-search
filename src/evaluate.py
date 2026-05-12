"""Evaluate three configs over the manual eval set.

Configs:
  A: CLIP image embeddings + FAISS, no re-rank   (default index built from images)
  B: CLIP text  embeddings + FAISS, no re-rank   (uses a parallel index built from titles)
  C: CLIP image embeddings + FAISS + GPT-4o re-rank

Metrics: Recall@K, Answer F1 (token overlap), latency (mean, p95).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src import retrieval, rerank as rr

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "eval" / "eval_set.json"
RESULTS = ROOT / "data" / "eval" / "results.json"
CATALOG_CSV = ROOT / "data" / "processed" / "catalog.csv"
EMB_DIR = ROOT / "data" / "embeddings"
TEXT_INDEX = EMB_DIR / "faiss_text.index"
TEXT_ID_MAP = EMB_DIR / "id_map_text.json"

MODEL_NAME = "openai/clip-vit-base-patch32"


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def build_text_index() -> None:
    """One-time: encode catalog titles+descriptions with CLIP text encoder, build a FAISS index."""
    df = pd.read_csv(CATALOG_CSV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_NAME)

    feats_all: list[np.ndarray] = []
    id_map: dict[int, str] = {}
    BATCH = 64
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH), desc="text-embed"):
            batch = df.iloc[start : start + BATCH]
            texts = [
                f"{r['title']}. {str(r.get('description',''))[:200]}"
                for _, r in batch.iterrows()
            ]
            inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            f = model.get_text_features(**inputs)
            if not torch.is_tensor(f):
                f = getattr(f, "text_embeds", None) or f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            feats_all.append(f.cpu().numpy().astype(np.float32))
            for offset, (_, row) in enumerate(batch.iterrows()):
                id_map[len(id_map)] = str(row["product_id"])
    arr = np.concatenate(feats_all, axis=0)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    faiss.write_index(index, str(TEXT_INDEX))
    TEXT_ID_MAP.write_text(json.dumps(id_map))
    print(f"[eval] built text index {arr.shape} -> {TEXT_INDEX}")


def _load_text_index():
    if not TEXT_INDEX.exists():
        build_text_index()
    return faiss.read_index(str(TEXT_INDEX)), {
        int(k): v for k, v in json.loads(TEXT_ID_MAP.read_text()).items()
    }


def _search_text_index(query: str, k: int) -> list[dict]:
    index, id_map = _load_text_index()
    vec = retrieval.encode_text(query)
    scores, idxs = index.search(vec, k)
    catalog = retrieval._catalog()
    out: list[dict] = []
    for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        pid = id_map.get(i)
        if pid is None or pid not in catalog.index:
            continue
        row = catalog.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        out.append({
            "product_id": pid,
            "title": row["title"],
            "description": row.get("description", ""),
            "color": row.get("color", ""),
            "image_path": row["image_path"],
            "score": float(s),
        })
    return out


RECALL_KS = (1, 3, 5)


def run_config(config: str, eval_items: list[dict], k: int) -> dict:
    hits: dict[int, int] = {kk: 0 for kk in RECALL_KS if kk <= k}
    f1s: list[float] = []
    latencies: list[float] = []

    for item in tqdm(eval_items, desc=f"config {config}"):
        q = item["question"]
        gt_id = str(item["product_id"])
        gt_ans = item.get("ground_truth_answer", "")

        t0 = time.time()
        if config == "A":
            results = retrieval.search_text(q, k)
        elif config == "B":
            results = _search_text_index(q, k)
        elif config == "C":
            results = retrieval.search_text(q, k)
            results = rr.rerank(q, results, top_n=k)
        else:
            raise ValueError(config)
        latencies.append((time.time() - t0) * 1000.0)

        ids = [str(r["product_id"]) for r in results]
        for kk in hits:
            if gt_id in ids[:kk]:
                hits[kk] += 1

        if config == "C" and results:
            ans = rr.answer(q, results[0])
        else:
            ans = results[0]["title"] if results else ""
        f1s.append(f1(ans, gt_ans))

    n = max(1, len(eval_items))
    out: dict = {"config": config, "n": len(eval_items)}
    for kk, c in hits.items():
        out[f"recall@{kk}"] = c / n
    out["answer_f1"] = statistics.mean(f1s) if f1s else 0.0
    out["latency_ms_mean"] = statistics.mean(latencies) if latencies else 0.0
    out["latency_ms_p95"] = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20 else max(latencies, default=0.0)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--limit", type=int, default=0, help="Limit eval items (0 = all)")
    args = parser.parse_args()

    items = json.loads(EVAL_SET.read_text())
    if args.limit:
        items = items[: args.limit]

    out = []
    for cfg in args.configs.split(","):
        cfg = cfg.strip()
        if not cfg:
            continue
        out.append(run_config(cfg, items, args.k))

    RESULTS.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[eval] wrote {RESULTS}")


if __name__ == "__main__":
    main()

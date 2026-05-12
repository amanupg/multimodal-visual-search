"""Draft an eval set with GPT-4o from the catalog metadata.

For each sampled product, GPT-4o is asked to write ONE natural-language question
that the product uniquely answers, plus a one-sentence ground-truth answer.

Output: data/eval/eval_set.json with up to N entries. The PRD asks for ~80
manually verified pairs — drafting 150 with GPT-4o, then trimming/editing by
hand is the recommended workflow.

Usage:
    python -m scripts.build_eval_set --n 150
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "processed" / "catalog.csv"
EVAL_SET = ROOT / "data" / "eval" / "eval_set.json"

MODEL = "gpt-4o"

PROMPT = """You write evaluation questions for a product search system.

Given the product below, write ONE natural-language search query a real shopper
might type that the product uniquely matches, and a one-sentence ground-truth
answer that names the product.

Return strict JSON only: {{"question": "...", "ground_truth_answer": "..."}}

Product:
title: {title}
category: {category}
color: {color}
description: {description}
"""


def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(CATALOG_CSV)
    random.seed(args.seed)
    sample = df.sample(min(args.n, len(df)), random_state=args.seed)

    client = _client()
    out: list[dict] = []
    for _, row in tqdm(list(sample.iterrows()), desc="draft"):
        prompt = PROMPT.format(
            title=row["title"],
            category=row.get("category", ""),
            color=row.get("color", ""),
            description=str(row.get("description", ""))[:400],
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            text = (resp.choices[0].message.content or "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                continue
            obj = json.loads(text[start : end + 1])
            obj["product_id"] = str(row["product_id"])
            out.append(obj)
        except Exception as e:  # noqa: BLE001
            print(f"[eval-draft] skip {row['product_id']}: {e}")

    EVAL_SET.write_text(json.dumps(out, indent=2))
    print(f"[eval-draft] wrote {len(out)} candidates -> {EVAL_SET}")
    print("Manually review and trim to ~80 verified pairs.")


if __name__ == "__main__":
    main()

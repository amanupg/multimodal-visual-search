"""GPT-4o re-ranking + 1-sentence grounded answer.

Sends candidate product images + metadata to GPT-4o, gets relevance scores,
re-sorts and produces a one-sentence answer using the top product.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parents[1]
MAX_SIDE = 512
MODEL = "gpt-4o"


def _client():
    from openai import OpenAI  # lazy import so retrieval works without the key
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _img_to_data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_rerank_messages(query: str, candidates: list[dict]) -> list[dict]:
    text_parts = [
        "You are a product search assistant.",
        f'User query: "{query}"',
        f"Below are {len(candidates)} candidate products. Score each 0-10 for relevance.",
        "Return ONLY a JSON array of objects with keys product_id, score, reason.",
        "Example: [{\"product_id\": \"...\", \"score\": 8, \"reason\": \"...\"}]",
        "",
        "Candidates:",
    ]
    for i, c in enumerate(candidates, 1):
        meta = f"{i}. product_id={c['product_id']} | title={c['title']}"
        if c.get("color"):
            meta += f" | color={c['color']}"
        text_parts.append(meta)

    content: list[dict] = [{"type": "text", "text": "\n".join(text_parts)}]
    for c in candidates:
        path = ROOT / c["image_path"]
        if path.exists():
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(path)}})
    return [{"role": "user", "content": content}]


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """Re-rank candidates with GPT-4o; falls back to original order on error."""
    if not candidates:
        return []
    try:
        client = _client()
        messages = _build_rerank_messages(query, candidates)
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0, max_tokens=600
        )
        scored = _parse_json_array(resp.choices[0].message.content or "")
        score_by_id = {str(s.get("product_id")): s for s in scored if "product_id" in s}
        for c in candidates:
            s = score_by_id.get(str(c["product_id"]))
            c["rerank_score"] = float(s["score"]) if s and "score" in s else 0.0
            c["rerank_reason"] = s.get("reason", "") if s else ""
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]
    except Exception as e:  # noqa: BLE001
        print(f"[rerank] failed ({e}); returning original order")
        return candidates[:top_n]


def answer(query: str, top_product: dict) -> str:
    """One-sentence grounded answer using the top product's image + metadata."""
    try:
        client = _client()
        path = ROOT / top_product["image_path"]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f'User query: "{query}"\n'
                    f"Top product: {top_product['title']}\n"
                    "Write ONE concise sentence answering the user, grounded in the image and title. "
                    "Mention the product name."
                ),
            }
        ]
        if path.exists():
            content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(path)}})
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=80,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"Top match: {top_product['title']}. (answer generation skipped: {e})"

"""Streamlit UI for the multimodal visual search demo.

Run:
    streamlit run src/app.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from src import retrieval, rerank as rr

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Multimodal Visual Search", layout="wide")
st.title("Multimodal Visual Search — ABO Catalog")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-K", 1, 10, 5)
    rerank_on = st.toggle("GPT-4o re-rank + answer", value=False,
                           help="Requires OPENAI_API_KEY in .env")
    category_filter = st.text_input("Category filter (substring, optional)", "")

tab_text, tab_image = st.tabs(["Text search", "Image search"])


def _filter_results(results: list[dict], substr: str) -> list[dict]:
    if not substr:
        return results
    s = substr.lower()
    return [r for r in results if s in str(r.get("category", "")).lower()]


def _render_grid(results: list[dict]) -> None:
    cols = st.columns(min(5, max(1, len(results))))
    for i, r in enumerate(results):
        col = cols[i % len(cols)]
        with col:
            img_path = ROOT / r["image_path"]
            if img_path.exists():
                col.image(str(img_path), use_container_width=True)
            score = r.get("rerank_score", r["score"])
            label = "rerank" if "rerank_score" in r else "sim"
            col.markdown(f"**{r['title'][:60]}**")
            col.caption(f"{label}: {score:.3f}  ·  `{r['product_id']}`")
            if r.get("rerank_reason"):
                col.caption(r["rerank_reason"][:120])


with tab_text:
    query = st.text_input("Query", placeholder="e.g., a brown leather sofa with wooden legs")
    if st.button("Search", type="primary", disabled=not query.strip()):
        t0 = time.time()
        results = retrieval.search_text(query, k)
        results = _filter_results(results, category_filter)
        answer_text: str | None = None
        if rerank_on and results:
            results = rr.rerank(query, results, top_n=k)
            answer_text = rr.answer(query, results[0])
        elapsed_ms = (time.time() - t0) * 1000
        if answer_text:
            st.success(answer_text)
        if results:
            _render_grid(results)
        else:
            st.warning("No results.")
        st.caption(f"Query time: {elapsed_ms:.0f} ms")

with tab_image:
    upload = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png"])
    if upload is not None:
        img = Image.open(io.BytesIO(upload.read())).convert("RGB")
        st.image(img, caption="Reference", width=240)
        if st.button("Find similar", type="primary"):
            t0 = time.time()
            results = retrieval.search_image(img, k)
            results = _filter_results(results, category_filter)
            elapsed_ms = (time.time() - t0) * 1000
            if results:
                _render_grid(results)
            else:
                st.warning("No results.")
            st.caption(f"Query time: {elapsed_ms:.0f} ms")

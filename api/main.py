"""FastAPI backend exposing the search pipeline.

Run:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src import retrieval, rerank as rr

app = FastAPI(title="Multimodal Visual Search")


class TextSearchRequest(BaseModel):
    query: str
    k: int = 5
    rerank: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search/text")
def search_text(req: TextSearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(400, "query must be non-empty")
    results = retrieval.search_text(req.query, req.k)
    answer_text: str | None = None
    if req.rerank and results:
        results = rr.rerank(req.query, results, top_n=req.k)
        answer_text = rr.answer(req.query, results[0])
    return {"query": req.query, "k": req.k, "results": results, "answer": answer_text}


@app.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    k: int = Form(5),
) -> dict:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {e}")
    results = retrieval.search_image(img, k)
    return {"k": k, "results": results}

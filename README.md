# Multimodal Visual Search for Product Catalogs

**EECS 6895 — Advanced Big Data and AI · Columbia University · Spring 2026**
Aman Upganlawar (au2327)

End-to-end multimodal product search: query a 2,711-product furniture catalog
with **natural language or a reference image**, get back a ranked list of
visually relevant products plus a grounded one-sentence answer.
Built with **CLIP ViT-B/32** for cross-modal retrieval, **FAISS** for cosine
search, and **GPT-4o** as a vision-LLM re-ranker.

- **Demo video:** <https://drive.google.com/drive/folders/1UQDMY3QsPpgrMT2Xeb9V1IT2Nfmk9mXM?usp=sharing>
- **Final report (PDF):** `au2327.pdf` *(see CourseWorks submission)*
- **Presentation deck:** [`docs/au2327_Final_Project_Presentation.pptx`](docs/au2327_Final_Project_Presentation.pptx)

---

## Headline result

Three configurations, evaluated on the same 150 GPT-drafted questions over the
same 2,711-product furniture catalog:

| Config | Retriever | Re-rank | **R@1** | R@3 | R@5 | Answer F1 | Mean latency |
|---|---|---|---|---|---|---|---|
| **A** | CLIP image + FAISS | — | 0.113 | 0.207 | 0.253 | 0.424 | **58 ms** |
| **B** | CLIP text + FAISS | — | 0.173 | 0.327 | 0.400 | 0.373 | 1.07 s |
| **C** | CLIP image + FAISS | GPT-4o (top-5) | **0.233** | 0.253 | 0.253 | 0.430 | 4.43 s |

**The vision-LLM re-rank doubles top-1 recall (0.113 → 0.233, +12 pp)** while
preserving the same R@5 ceiling — i.e. it improves *ordering*, not *coverage*.
Raw numbers in [`data/eval/results.json`](data/eval/results.json).

---

## Repository structure

```
.
├── src/                      # core Python pipeline
│   ├── ingest.py             # stream ABO, filter to furniture, download images
│   ├── embed.py              # CLIP ViT-B/32 -> embeddings.npy
│   ├── index.py              # build FAISS IndexFlatIP (cosine)
│   ├── retrieval.py          # CLIP text/image -> FAISS top-K
│   ├── rerank.py             # GPT-4o re-rank + 1-sentence grounded answer
│   ├── evaluate.py           # three-config ablation harness
│   └── app.py                # Streamlit demo UI
├── api/
│   └── main.py               # FastAPI: /health, /search/text, /search/image
├── scripts/
│   ├── build_eval_set.py     # GPT-drafted candidate QA pairs
│   ├── smoke_rerank.py       # one-shot rerank test
│   └── smoke_api.py          # in-process FastAPI smoke test
├── data/
│   ├── processed/catalog.csv # 2,711 products: title, description, color, image_path
│   ├── eval/eval_set.json    # 150 question / ground-truth pairs
│   ├── eval/results.json     # the three-config numbers shown above
│   ├── embeddings/           # .npy + FAISS indexes (gitignored, regenerable)
│   └── raw/                  # ABO images + metadata (gitignored, regenerable)
├── demo/
│   ├── images/               # 5 reference photos for image-search demo
│   └── Final_demo_video.mp4  # gitignored, hosted externally (see top of README)
├── docs/                     # presentation deck and figures
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

Tested on Python 3.11, Windows + macOS, CPU only. No GPU required.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your OpenAI key (required only for re-rank + eval config C)
cp .env.example .env
# Edit .env and set:
#   OPENAI_API_KEY=sk-...
```

`.env` is gitignored. Never commit your key.

---

## How to run

### A. Reproduce the catalog and index (one-time, ~25 min on CPU)

```bash
# 1. Stream the Amazon Berkeley Objects metadata, filter to furniture,
#    download images, and write data/processed/catalog.csv.
python -m src.ingest --category furniture --limit 4000

# 2. Embed every product image with CLIP ViT-B/32.
#    Produces data/embeddings/embeddings.npy + id_map.json.
python -m src.embed

# 3. Build the FAISS IndexFlatIP cosine index.
python -m src.index
```

> Smoke test the ingest step first with `--limit 200` if you want to make sure
> the S3 fetch path works before committing to the full run.

### B. Try a query from the CLI

```bash
python -m src.retrieval --query "a brown leather sofa with wooden legs" -k 5
```

### C. Run the API

```bash
uvicorn api.main:app --reload
# Endpoints:
#   GET  /health
#   POST /search/text   {"query": "...", "k": 5, "rerank": true}
#   POST /search/image  multipart image upload
```

### D. Run the Streamlit demo UI (the one in the video)

```bash
streamlit run src/app.py
# Open http://localhost:8501
```

The UI has two tabs (text / image), a top-K slider, and a re-rank toggle. See
[`DEMO_QUERIES.md`](DEMO_QUERIES.md) for five suggested text queries and the
expected behaviour for each.

### E. Reproduce the three-config evaluation

```bash
# 1. (Optional) Re-draft eval candidates with GPT-4o. The shipped file is a
#    150-pair version that has already been manually curated.
python -m scripts.build_eval_set --n 150

# 2. Run all three configs:
#    A = CLIP image + FAISS
#    B = CLIP text  + FAISS
#    C = CLIP image + FAISS + GPT-4o re-rank
python -m src.evaluate -k 5
```

Results are written to `data/eval/results.json` and printed to stdout. Config C
requires `OPENAI_API_KEY`; configs A and B do not.

---

## Example usage

Single text query, no re-rank:

```python
from src import retrieval
hits = retrieval.search_text("a navy velvet tufted sectional sofa", k=5)
for h in hits:
    print(h["score"], h["title"])
```

Single text query, with GPT-4o re-rank:

```python
from src import retrieval, rerank
hits = retrieval.search_text("a navy velvet tufted sectional sofa", k=5)
ranked = rerank.rerank("a navy velvet tufted sectional sofa", hits, top_n=5)
answer = rerank.answer("a navy velvet tufted sectional sofa", ranked[0])
print(answer)
```

Image query:

```python
from src import retrieval
hits = retrieval.search_image("demo/images/1_navy_velvet_sectional.jpg", k=5)
```

---

## Dataset

This project uses the public, CC-BY-licensed **Amazon Berkeley Objects (ABO)**
catalog (Collins et al., CVPR 2022). I curated a 2,711-product furniture
subset with English-language metadata and 256-px product images. The full ABO
release is at:
<https://amazon-berkeley-objects.s3.amazonaws.com/index.html>

The 150-pair evaluation set in `data/eval/eval_set.json` was drafted by GPT-4o
from product metadata and then manually filtered.

---

## License & attribution

Code: MIT (see `LICENSE` if present).
ABO image and metadata copyright Amazon.com Inc. and the original photographers,
distributed under CC BY 4.0. CLIP weights from
[OpenAI / Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32).
GPT-4o is a service of OpenAI.

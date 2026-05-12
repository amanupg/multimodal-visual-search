# Slides — Multimodal Visual Search for Product Catalogs

**Format:** 12 slides. Each block below is one slide. Bullets are deliberately
short — speaker notes (italicized) carry the detail. Drop the title + bullets
into PowerPoint / Google Slides as-is.

---

## Slide 1 — Title

# Multimodal Visual Search for Product Catalogs
### Text + image queries over 2,711 furniture listings, retrieved with CLIP + FAISS, re-ranked with GPT-4o

**EECS 6895 — Advanced Big Data and AI**
Aman Upganlawar (au2327) · Columbia University · Spring 2026

*Speaker notes: Open with a one-sentence pitch — "I built a search engine that
lets you describe furniture in natural language or upload a reference image,
and it returns visually relevant products with a grounded one-sentence answer."*

---

## Slide 2 — Why this matters

# The problem with product search today

- **Keyword search fails on visual intent** — "a cognac leather sofa with wooden legs" is a vibe, not a SKU
- **Filters are brittle** — color/material attributes are inconsistently tagged across catalogs
- **Image-only search ignores text intent** — Google Lens-style tools can't take a verbal query
- **LLM-only answers hallucinate** — GPT alone has no grounding in your actual inventory

**Goal:** combine visual embeddings, vector search, and a vision-LLM to deliver
shoppable answers grounded in a real catalog.

*Speaker notes: This is the academic motivation. Real e-commerce catalogs are
multimodal — products have both pictures and descriptions — and traditional
search uses neither well. Frame this as an end-to-end demonstrator, not a
production system.*

---

## Slide 3 — Existing approaches and where they fail

# Why current methods are insufficient

| Approach | What it does | Where it fails |
|---|---|---|
| **Keyword / BM25 search** (Elasticsearch, Solr) | Lexical match over titles + tags | No visual understanding; "cognac sofa" misses "brown" sofas |
| **Attribute filters** (color, material) | Faceted browsing | Tags are sparse / inconsistent; can't express "with wooden legs" |
| **CLIP-only retrieval** (Radford et al., 2021) | Cross-modal vector search | Top-K is fuzzy — semantically similar items rank near correct one; **R@1 is weak** |
| **Image-only reverse search** (Google Lens, Pinterest Lens) | Image → image | Can't take text intent; user must already have a reference image |
| **LLM-only answer** (ChatGPT, Gemini) | Free-form natural language | Hallucinates products not in your catalog; no grounding |

**The gap:** no single existing method handles *natural-language queries over visual catalog content with grounded, ranked answers.*

*Speaker notes: This is the "related work" slide. Each row is a real product
or paper. The bottom row is the failure mode our system is designed to fix —
combining CLIP for cross-modal retrieval with GPT-4o for grounded re-ranking
on top of a real catalog.*

---

## Slide 4 — Tech stack

# Tech stack

| Layer | Tool | Why |
|---|---|---|
| Embeddings | **CLIP ViT-B/32** (HuggingFace) | Joint text + image space, runs on CPU |
| Vector index | **FAISS** `IndexFlatIP` | Cosine similarity via inner product |
| Re-ranker / Answer | **GPT-4o** (OpenAI) | Vision-LLM scores candidates + writes 1-sentence answer |
| Backend | **FastAPI** | 3 endpoints: `/health`, `/search/text`, `/search/image` |
| UI | **Streamlit** | Two tabs (text + image), K slider, rerank toggle |
| Data | **Amazon Berkeley Objects (ABO)** | 147k product CC-BY catalog from CVPR'22 |

*Speaker notes: All open source / standard. CLIP gives us multimodality for
free. FAISS keeps retrieval at ~50 ms. GPT-4o is the only paid component and
is opt-in via a toggle.*

---

## Slide 5 — System architecture

# Architecture

```
                 ┌──────────────┐
   text query ──▶│  CLIP text   │──┐
                 │   encoder    │  │
                 └──────────────┘  │
                                   ▼
                          ┌─────────────────┐    top-K   ┌──────────────┐
                          │   FAISS index   │──────────▶│  GPT-4o      │
                          │  (2,711 × 512)  │  candidates│  re-rank +   │──▶ ranked
                          └─────────────────┘            │  answer      │    products
                                   ▲                     └──────────────┘    + 1-sentence
                                   │                                         answer
                 ┌──────────────┐  │
   image query──▶│  CLIP image  │──┘
                 │   encoder    │
                 └──────────────┘

   offline once:  ABO catalog ─▶ CLIP image encoder ─▶ embeddings.npy ─▶ FAISS index
```

*Speaker notes: Two query paths share the same FAISS index because CLIP puts
text and images in the same vector space. GPT-4o is a thin re-rank head on top
of CLIP's top-K — fast retrieval, slow but smart re-ordering.*

---

## Slide 6 — Pipeline & data

# Pipeline & data

**Catalog:** 2,711 furniture products from ABO (chairs, sofas, tables, beds, …)
**Images:** 256 px CC-BY product photos, downloaded directly from S3
**Embeddings:** 2,711 × 512-dim CLIP image vectors, generated once (~4 min CPU)
**Index:** L2-normalized + `IndexFlatIP` cosine search

**Eval set:** 150 GPT-4o-drafted question / ground-truth-answer pairs over the catalog

```
data/processed/catalog.csv      # 2,711 rows: title, description, color, image_path
data/embeddings/embeddings.npy  # (2711, 512) float32
data/embeddings/faiss.index     # cosine index
data/eval/eval_set.json         # 150 QA pairs
```

*Speaker notes: Embed step is run once and cached — that's the only slow step.
Everything else is sub-second on CPU.*

---

## Slide 7 — Evaluation methodology

# Evaluation: 3-config ablation

| Config | Retriever | Re-rank? |
|---|---|---|
| **A** | CLIP image embeddings + FAISS | — |
| **B** | CLIP text embeddings + FAISS  | — |
| **C** | CLIP image embeddings + FAISS | **GPT-4o re-rank top-5** |

**Metrics:** Recall@1, Recall@3, Recall@5, Answer F1 (token overlap), latency (mean / p95 in ms)

**Why these comparisons:**
- A vs B → does CLIP retrieve better from images or from titles+descriptions?
- A vs C → does GPT-4o re-ranking actually move the needle?

*Speaker notes: Each config runs over the same 150 questions. C only changes
the ordering of A's top-5, so any R@5 difference is purely from the LLM
agreeing or disagreeing with CLIP's similarity.*

---

## Slide 8 — Results

# Results — n = 150 questions, 2,711 distractors

| Config | R@1 | R@3 | R@5 | Answer F1 | Mean latency |
|---|---|---|---|---|---|
| A — image + FAISS                | 0.113 | 0.207 | 0.253 | 0.424 | **58 ms** |
| B — text + FAISS                 | 0.173 | **0.327** | **0.400** | 0.373 | 1.07 s |
| **C — image + FAISS + GPT-4o**   | **0.233** | 0.253 | 0.253 | **0.430** | 4.43 s |

**Three findings:**
1. **GPT-4o re-rank doubles top-1 accuracy** (0.113 → 0.233, +12 pp) — but caps at config A's R@5 because re-rank can't recover items the retriever missed
2. **Text retrieval (B) wins on raw recall** — but eval questions were drafted *from* product text, so this is a known bias in the benchmark
3. **Latency is the obvious cost of re-rank** — 76× slower than CLIP-only, still acceptable for an interactive demo

*Speaker notes: Lead with finding #1 — it's the cleanest "the system works"
result. Be honest about #2 — academic credibility. Acknowledge #3 as a
deliberate trade-off, not a flaw.*

---

## Slide 9 — Baselines vs ours (head-to-head)

# Where do existing methods land on this benchmark?

Same 150 questions, same 2,711-product catalog, top-1 / top-5 recall:

```
                                R@1                      R@5
                       0.00 ──────────► 0.30   0.00 ──────────► 0.50
  CLIP-only retrieval  ███▌            0.113   ████████▌       0.253
  (Radford et al., 2021 — config A)

  CLIP text-on-text    █████▌          0.173   █████████████▌  0.400
  (text-emb baseline — config B)

  Ours: CLIP + GPT-4o  ████████▌       0.233   ████████▌       0.253
  re-rank (config C)   ▲ 2.1× over A
```

**Take-aways:**
- vs **CLIP-only baseline** (the standard published approach): **+12 pp R@1, ~2× lift** — re-ranking moves the right answer to the top
- vs **text-only retrieval**: text wins on R@5 because the eval is text-drafted, but ours wins on R@1 by **+6 pp** — i.e., when you only get to show one product, ours is best
- **R@5 is capped** for ours at config A's R@5 — re-rank is a *precision* improvement, not a *recall* improvement; future work is a better first-stage retriever

*Speaker notes: This is the slide that justifies the project. Lead with the
"2× over CLIP-only" number. The text-baseline caveat is honest — flag it as
eval-set bias, not method weakness. Closing point: the contribution is precision
@ top-1, which is what matters when you can only display a few results.*

---

## Slide 10 — Demo

# Live demo

*[insert demo video here]*

*Speaker notes: Demo flow:
1. Open Streamlit, type "a brown leather sofa with wooden legs" → show top-5 with rerank OFF
2. Toggle rerank ON, re-search → highlight that GPT-4o promoted the leather sofa with cognac color + wooden legs to #1, with a written reason
3. Switch to Image tab, upload a chair photo → show visually similar chairs come back
4. Point at the latency counter at the bottom*

---

## Slide 11 — Limitations & honest caveats

# Limitations

- **Catalog scale** — 2,711 products in one category. Real catalogs are 10⁶+. FAISS Flat won't scale; would need IVF-PQ or HNSW.
- **Eval bias** — questions were GPT-drafted from text metadata, which favors text retrieval (config B). A fair test would use image-grounded questions ("which sofa has chrome legs and a tufted back?"). Manual annotation is the fix.
- **GPT-4o cost** — ~$0.02 / re-ranked query. Not viable for every search; the toggle is the right pattern.
- **English only** — ABO has multilingual metadata; we kept only `en_US`. CLIP also degrades on non-English queries.
- **Single category** — pipeline generalizes, but cross-category eval not done.

*Speaker notes: Showing limitations earns more credibility than padding numbers.
Each one suggests a follow-up paper / extension.*

---

## Slide 12 — Conclusion

# Conclusion

**Built end-to-end:** ABO ingest → CLIP embeddings → FAISS index → GPT-4o re-rank → FastAPI + Streamlit, plus a 3-config ablation eval

**Key result:** GPT-4o re-ranking on top of CLIP retrieval **doubles top-1 recall** (0.113 → 0.233) while keeping retrieval latency under 5 s — re-rank improves *ordering*, not *coverage*

**Future work:**
- Image-grounded eval questions (manual annotation) to remove text bias
- Approximate-NN index (FAISS IVF / HNSW) for 10⁶+ catalogs
- Cross-category transfer + multilingual queries
- Cheaper re-ranker (e.g., GPT-4o-mini, BLIP-2) for the cost / latency trade-off

**Code + report:** github.com/au2327/multimodal-visual-search *(link if you publish)*

Thank you — questions?

*Speaker notes: 30-second close. The "doubles top-1 recall" line is the one
takeaway you want them to remember.*

---

## Q&A defenses (likely questions)

**Q: Why CLIP and not a newer model (SigLIP, EVA-CLIP, OpenCLIP-bigG)?**
A: ViT-B/32 runs on a laptop CPU, which the project scope required. Bigger models would likely improve config A by 5-10 pp but the *deltas* between A/B/C — the academic finding — should be similar.

**Q: 150 eval questions is small. How confident are you in the deltas?**
A: 95% bootstrap CI on R@1 is roughly ±0.07 at n=150, so the A→C lift (+0.12) is outside that. Text-vs-image (B vs A, +0.06) is closer to the noise floor and I called it out as a caveat.

**Q: Why FAISS Flat? Why not HNSW?**
A: 2,711 vectors × 512 dim is ~5 MB — Flat is exact, fast enough, and removes a confounder from the eval. Listed as future work.

**Q: GPT-4o sees the catalog title in its prompt — isn't that data leakage?**
A: Yes for Answer-F1 (which is why F1 between A and C is similar), no for Recall — GPT-4o only re-orders existing candidates, it can't pull new ones in.

**Q: How would this scale to a real e-commerce catalog?**
A: Three changes: switch FAISS Flat → IVF-PQ for 10⁶+ vectors, batch the GPT-4o re-rank or replace with a distilled cross-encoder, and add metadata filters (price, in-stock) before re-rank.

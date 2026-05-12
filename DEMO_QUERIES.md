# Demo cheat sheet

Streamlit URL: **http://localhost:8501**
Pre-prepared images: `demo/images/` (five .jpg files)

---

## 5 text queries that show the system off

Use these in the **Text search** tab. For each one, run **once with rerank
OFF**, then **once with rerank ON**, so the audience sees the lift.

### 1. "a brown leather sofa with wooden legs"
- *No rerank:* CLIP returns 5 leather sofas, all close in score (~0.34). Top-1 may or may not have wooden legs.
- *Rerank ON:* GPT-4o promotes the Rivet Alonzo cognac sofa with wooden legs to #1 with a written reason.
- **Talking point:** rerank reads visual fine-detail (legs, color shade) that CLIP averages over.

### 2. "a navy velvet tufted sectional sofa"
- *No rerank:* should hit the Rivet Eva and similar tufted navy/dark velvet sofas.
- *Rerank ON:* GPT-4o explicitly checks for "tufting" and "sectional" shape.
- **Talking point:** combining material + color + shape constraints is what CLIP+rerank does well together.

### 3. "a modern walnut side table with black metal hairpin legs"
- *No rerank:* CLIP retrieves several walnut/black side tables.
- *Rerank ON:* the Rivet Bristol with literal hairpin legs should win.
- **Talking point:** specific design vocabulary ("hairpin legs") is where rerank shines.

### 4. "a high-back executive office chair in black leather"
- *No rerank:* multiple AmazonBasics executive chairs come back tightly clustered.
- *Rerank ON:* the highest-back, leather variant gets bumped.
- **Talking point:** when many candidates are near-duplicates, ordering matters.

### 5. "a comfortable mid-century accent chair for a reading nook"
- *No rerank:* fuzzy / vibe-based query — CLIP returns a mix.
- *Rerank ON:* GPT-4o has to actually reason about "comfortable" + "mid-century" + use-case.
- **Talking point:** this is the query type traditional search fails on most. It's the strongest motivation slide-3 example.

---

## 5 image uploads in `demo/images/`

Use these in the **Image search** tab. They're all real catalog products, so:
- Top-1 should be the *exact same product* (sanity check that retrieval works)
- Top-2 through top-5 should be visually similar items — this is the interesting demo

| File | What it is | What you'll get back |
|---|---|---|
| `1_navy_velvet_sectional.jpg` | Rivet Eva navy velvet tufted sectional | Other tufted velvet sofas / sectionals |
| `2_tufted_queen_bed.jpg` | Stone & Beam Tisbury upholstered queen bed | Other upholstered platform beds |
| `3_walnut_side_table.jpg` | Rivet Bristol walnut + black-metal side table | Other small wood/metal side tables |
| `4_round_leather_ottoman.jpg` | Stone & Beam Norah round leather ottoman | Other ottomans / round footstools |
| `5_black_office_chair.jpg` | AmazonBasics black task office chair | Other office / task / executive chairs |

**Demo flow for image search:**
1. Upload `4_round_leather_ottoman.jpg` (most visually distinct of the five).
2. Show the top-5 grid — point out that result #1 is the exact item, and #2-#5 are visually similar (round, brown, leather, ottoman-shaped).
3. Note query latency at the bottom (~50 ms).

---

## "Phone photo" demo (optional but powerful)

If you want to demo with something *outside* the catalog: take a phone photo
of any chair / table / sofa in your apartment, upload it, and show the system
finds visually similar products from the ABO catalog. This is the strongest
visceral demo — works because CLIP image embeddings generalize.

---

## Demo script (in order)

**~3 minutes total**

1. **(20s)** Open Streamlit, sidebar visible. Set K=5, rerank OFF.
2. **(30s)** Text query #1 (`brown leather sofa with wooden legs`), rerank OFF. Read top-3 titles aloud. Note query time (~50 ms).
3. **(30s)** Toggle rerank ON, re-search same query. Show how the top-1 changed and the *reason* now appears under each card. Note new query time (~4 s) — call out the latency trade-off.
4. **(40s)** Switch to Image tab. Upload `4_round_leather_ottoman.jpg`. Show top-5 grid: #1 is the exact item, others are visually similar.
5. **(30s)** (Optional) Upload phone photo of your couch / chair / desk. Show generalization.
6. **(30s)** Pivot back to slides → results table.

---

## Backup queries (if any of the above don't land well)

- "a small velvet ottoman in pink or blush"
- "a metal-framed dining chair with a leather seat"
- "a low platform bed with grey upholstered headboard"
- "a glass-top coffee table with chrome legs"
- "a wingback armchair in green or emerald"

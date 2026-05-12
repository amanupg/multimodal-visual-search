"""Single-query GPT-4o rerank smoke test."""
from src import retrieval, rerank as rr

QUERY = "a brown leather sofa with wooden legs"
K = 5
print(f"Query: {QUERY!r}")

base = retrieval.search_text(QUERY, K)
print("\nFAISS-only top-5:")
for r in base:
    print(f"  {r['score']:.3f}  {r['title'][:70]}")

print("\nGPT-4o re-ranked top-3:")
ranked = rr.rerank(QUERY, base, top_n=3)
for r in ranked:
    print(f"  rerank={r.get('rerank_score', 0):.1f}  {r['title'][:70]}")
    if r.get("rerank_reason"):
        print(f"     reason: {r['rerank_reason'][:140]}")

if ranked:
    print("\nGrounded answer:")
    print("  " + rr.answer(QUERY, ranked[0]))

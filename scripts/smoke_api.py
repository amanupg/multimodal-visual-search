"""In-process smoke test for the FastAPI app."""
from fastapi.testclient import TestClient
from api.main import app

c = TestClient(app)
print("health:", c.get("/health").json())
r = c.post("/search/text", json={"query": "leather sofa", "k": 3, "rerank": False})
j = r.json()
print(f"hits: {len(j['results'])}")
for x in j["results"]:
    print(f"  {x['score']:.3f}  {x['title'][:60]}")

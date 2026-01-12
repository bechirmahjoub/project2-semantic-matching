from fastapi import FastAPI, Query
from app.search import HybridSearchService
from app.schemas import SearchResponse

app = FastAPI(title="Project 2 - Semantic Matching", version="1.1.0")

service = HybridSearchService(
    equipment_csv="data/raw/equipment.csv",
    index_path="models/faiss.index",
    id_map_path="models/id_map.json",
    threshold=0.75,
)

@app.get("/")
def home():
    return {"message": "API is running. Use /health or /search?query=bosch%20industrial%20screwdriver&k=5"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    results, fallback_used = service.search(query=query, k=k)
    return {
        "query": query,
        "k": k,
        "threshold": service.threshold,
        "fallback_used": fallback_used,
        "results": results,
    }

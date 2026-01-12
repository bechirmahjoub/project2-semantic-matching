from fastapi import FastAPI, Query
from app.search import SemanticSearchService
from app.schemas import SearchResponse

app = FastAPI(title="Project 2 - Semantic Matching", version="1.0.0")

service = SemanticSearchService(
    equipment_csv="data/raw/equipment.csv",
    index_path="models/faiss.index",
    id_map_path="models/id_map.json",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    results = service.search(query=query, k=k)
    return {"query": query, "k": k, "results": results}

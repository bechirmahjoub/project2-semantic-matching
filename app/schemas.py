from pydantic import BaseModel
from typing import List

class SearchItem(BaseModel):
    item_id: str
    name: str
    description: str
    score: float

class SearchResponse(BaseModel):
    query: str
    k: int
    results: List[SearchItem]

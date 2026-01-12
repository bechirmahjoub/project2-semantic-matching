from pydantic import BaseModel
from typing import List, Literal, Optional

class SearchItem(BaseModel):
    item_id: str
    name: str
    description: str
    score: float                 # similarity or lexical score (for transparency)
    confidence: float            # calibrated [0,1]
    eligible: bool               # confidence >= threshold
    source: Literal["semantic", "lexical"]
    explanation: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    k: int
    threshold: float
    fallback_used: bool
    results: List[SearchItem]

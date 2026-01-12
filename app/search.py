import json
import math
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from app.lexical import TfidfLexicalSearch
from app.text_utils import normalize_text

def sigmoid_confidence(sim: float, a: float = 10.0, b: float = 0.45) -> float:
    """
    Map similarity -> confidence in [0,1] using a sigmoid.
    a: steepness, b: midpoint.
    """
    return float(1.0 / (1.0 + math.exp(-a * (sim - b))))

def short_explanation(query: str, name: str, desc: str) -> str:
    q_tokens = set(normalize_text(query).split())
    t_tokens = set(normalize_text(name + " " + desc).split())
    overlap = sorted(q_tokens & t_tokens)
    if overlap:
        return "Token overlap: " + ", ".join(overlap[:6])
    return "Semantic similarity match"

class HybridSearchService:
    def __init__(
        self,
        equipment_csv: str,
        index_path: str,
        id_map_path: str,
        threshold: float = 0.75,
    ):
        self.threshold = float(threshold)

        self.equipment = pd.read_csv(equipment_csv)
        self.equipment_map = {r["item_id"]: r for _, r in self.equipment.iterrows()}

        self.index = faiss.read_index(index_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.ids = meta["ids"]
        self.model = SentenceTransformer(meta["model"])

        # Lexical fallback
        self.lexical = TfidfLexicalSearch(self.equipment)

    def semantic_search(self, query: str, k: int):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, nn = self.index.search(q_emb, k)

        out = []
        for score, idx in zip(scores[0].tolist(), nn[0].tolist()):
            item_id = self.ids[idx]
            row = self.equipment_map[item_id]
            sim = float(score)
            conf = sigmoid_confidence(sim)
            out.append({
                "item_id": item_id,
                "name": str(row["name"]),
                "description": str(row["description"]),
                "score": sim,
                "confidence": conf,
                "eligible": conf >= self.threshold,
                "source": "semantic",
                "explanation": short_explanation(query, str(row["name"]), str(row["description"])),
            })
        return out

    def search(self, query: str, k: int = 5):
        semantic = self.semantic_search(query, k)
        fallback_used = False

        best_conf = semantic[0]["confidence"] if semantic else 0.0

        # Trigger fallback if semantic confidence is low
        if best_conf < self.threshold:
            fallback_used = True
            lex = self.lexical.search(query, k)

            lex_items = []
            for r in lex:
                # Lexical score is already in [0,1] for cosine similarity, but typically lower.
                # Map it to a conservative confidence band.
                lex_score = float(r["lexical_score"])
                conf = float(min(0.90, 0.50 + lex_score))
                lex_items.append({
                    "item_id": r["item_id"],
                    "name": r["name"],
                    "description": r["description"],
                    "score": lex_score,
                    "confidence": conf,
                    "eligible": conf >= self.threshold,
                    "source": "lexical",
                    "explanation": "Lexical fallback (TF-IDF cosine similarity)",
                })

            # Merge results by item_id (keep semantic if exists)
            by_id = {it["item_id"]: it for it in semantic}
            for it in lex_items:
                by_id.setdefault(it["item_id"], it)

            merged = sorted(by_id.values(), key=lambda x: (x["confidence"], x["score"]), reverse=True)
            return merged[:k], fallback_used

        return semantic, fallback_used

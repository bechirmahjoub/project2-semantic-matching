import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

class SemanticSearchService:
    def __init__(self, equipment_csv: str, index_path: str, id_map_path: str):
        self.equipment = pd.read_csv(equipment_csv)
        self.equipment_map = {r["item_id"]: r for _, r in self.equipment.iterrows()}

        self.index = faiss.read_index(index_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.ids = meta["ids"]
        self.model = SentenceTransformer(meta["model"])

    def search(self, query: str, k: int = 5):
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        scores, nn = self.index.search(q_emb, k)
        out = []
        for score, idx in zip(scores[0].tolist(), nn[0].tolist()):
            item_id = self.ids[idx]
            row = self.equipment_map[item_id]
            out.append({
                "item_id": item_id,
                "name": str(row["name"]),
                "description": str(row["description"]),
                "score": float(score),
            })
        return out

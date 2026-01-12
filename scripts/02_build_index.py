import json
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv("data/raw/equipment.csv")

    texts = (df["name"].fillna("") + " | " + df["description"].fillna("")).tolist()
    ids = df["item_id"].tolist()

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, "models/faiss.index")
    with open("models/id_map.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "ids": ids}, f, ensure_ascii=False, indent=2)

    print("Saved models/faiss.index and models/id_map.json")

if __name__ == "__main__":
    main()

import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def recall_at_k(ranks, k):
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))

def mrr(ranks):
    return float(np.mean([1.0 / r for r in ranks]))

def main():
    queries = pd.read_csv("data/raw/queries.csv")

    index = faiss.read_index("models/faiss.index")
    with open("models/id_map.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    ids = meta["ids"]
    model = SentenceTransformer(meta["model"])

    q_emb = model.encode(queries["query"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    q_emb = np.array(q_emb, dtype="float32")

    k = 10
    scores, nn = index.search(q_emb, k)

    ranks = []
    for i, true_id in enumerate(queries["true_item_id"].tolist()):
        retrieved = [ids[j] for j in nn[i]]
        if true_id in retrieved:
            ranks.append(retrieved.index(true_id) + 1)
        else:
            ranks.append(10_000)

    metrics = {
        "Recall@1": recall_at_k(ranks, 1),
        "Recall@3": recall_at_k(ranks, 3),
        "Recall@5": recall_at_k(ranks, 5),
        "Recall@10": recall_at_k(ranks, 10),
        "MRR": mrr([r for r in ranks if r < 10_000]) if any(r < 10_000 for r in ranks) else 0.0,
        "N": len(ranks),
    }
    print(metrics)

if __name__ == "__main__":
    main()

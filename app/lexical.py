from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.text_utils import normalize_text

class TfidfLexicalSearch:
    def __init__(self, equipment_df):
        self.df = equipment_df.copy()
        self.df["doc"] = (self.df["name"].fillna("") + " | " + self.df["description"].fillna("")).map(normalize_text)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X = self.vectorizer.fit_transform(self.df["doc"].tolist())

    def search(self, query: str, k: int = 5):
        q = normalize_text(query)
        Q = self.vectorizer.transform([q])
        sims = cosine_similarity(Q, self.X)[0]
        top_idx = sims.argsort()[::-1][:k]

        out = []
        for idx in top_idx:
            row = self.df.iloc[int(idx)]
            out.append({
                "item_id": str(row["item_id"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "lexical_score": float(sims[int(idx)]),
            })
        return out

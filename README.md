cat > README.md <<'EOF'
# Project 2 — NLP & Semantic Matching (Equipment Eligibility)

This project matches an extracted equipment designation (e.g., “Bosch industrial screwdriver”) against an equipment catalogue and returns top matches with scores, a calibrated confidence, an eligibility decision, and a brief explanation.

## Objectives (per assignment)
- Build a semantic index of eligible equipment (Sentence-Transformers + FAISS).
- Embed the input designation and compute similarity scores.
- Return top matches with confidence and a brief explanation.
- Provide fallback logic (lexical search) when semantic confidence is low.
- Provide evaluation metrics (Recall@k, MRR) in a notebook.

## Repository structure
- `app/` — FastAPI app + hybrid retrieval logic
- `scripts/` — data generation, indexing, evaluation
- `data/raw/` — equipment catalogue + query set (CSV)
- `models/` — FAISS index + id mapping
- `notebooks/` — evaluation notebook (`evaluation.ipynb`)
- `report.md` — results + interpretation

## Data
- Equipment catalogue: `data/raw/equipment.csv` with columns: `item_id`, `name`, `description`
- Query set: `data/raw/queries.csv` with columns: `query`, `true_item_id`

Synthetic data is generated via `scripts/01_make_synthetic_data.py`.

## Method overview

### Semantic retrieval (SBERT + FAISS)
1. Encode equipment documents with a Sentence-Transformers model.
2. Build a FAISS index on normalized embeddings (cosine similarity via inner product).
3. Embed the query and retrieve top-k nearest neighbours.

### Lexical fallback (TF-IDF cosine similarity)
If the top semantic confidence is below the threshold, the system also performs TF-IDF retrieval on the same documents and merges results.

### Text normalization
Text is normalized (lowercase, accent removal, punctuation removal, whitespace normalization) to reduce noise.
Implementation: `app/text_utils.py`.

### Confidence + thresholding + fallback logic
- The API returns both a raw similarity score (`score`) and a calibrated `confidence` in [0, 1].
- Eligibility decision:
  - `eligible = (confidence >= 0.75)`
- Fallback trigger:
  - If best semantic confidence `< 0.75`, run lexical TF-IDF fallback and include those results (`fallback_used = true`).

## Run locally (Python)

Create environment and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

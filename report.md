Project 2 — NLP & Semantic Matching (Equipment Eligibility)
1. Introduction

This project implements a semantic matching system that maps extracted equipment designations (e.g., “Bosch industrial screwdriver”) to an official catalogue of eligible equipment. The system returns the top-k matches, similarity scores, calibrated confidence values, and an eligibility decision based on a predefined threshold. When semantic confidence is low, a lexical fallback using TF-IDF is applied to improve robustness.

The system is designed to support noisy, incomplete, or paraphrased input queries while maintaining high recall and interpretability.

2. System Overview

The pipeline consists of the following components:

Text normalization

Lowercasing

Unicode normalization

Accent removal

Punctuation removal

Whitespace normalization

Semantic retrieval (Primary)

Sentence embeddings using a Sentence-Transformers model

FAISS approximate nearest-neighbor index

Cosine similarity (via normalized dot product)

Lexical fallback (Secondary)

TF-IDF vectorization

Cosine similarity over sparse vectors

Triggered when semantic confidence is below threshold

Confidence calibration and decision rule

Similarity scores are mapped to confidence values using a sigmoid function

A global threshold of 0.75 is used

If confidence ≥ 0.75 → Eligible

Otherwise → Not eligible + fallback may be applied

FastAPI endpoint

Exposes the full pipeline as a REST API

Returns structured JSON with explanations

3. Dataset

A synthetic dataset is used for development and evaluation.

3.1 Equipment Catalogue

Stored in: data/raw/equipment.csv
Fields:

item_id

name

description

3.2 Query Set

Stored in: data/raw/queries.csv
Fields:

query

true_item_id

The dataset includes paraphrases, abbreviations, and noisy surface forms to simulate real-world extraction errors.

4. Evaluation Methodology

The system is evaluated using standard information retrieval metrics:

Recall@k (k ∈ {1, 3, 5, 10})

Mean Reciprocal Rank (MRR)

Two systems are evaluated:

Semantic Retrieval (SBERT + FAISS)

Lexical Baseline (TF-IDF)

Evaluation scripts:

scripts/03_evaluate.py (semantic)

scripts/04_evaluate_tfidf.py (lexical)

5. Results

Evaluation was performed on N = 250 synthetic queries.

5.1 Semantic Retrieval (SBERT + FAISS)

Recall@1: 0.920

Recall@3: 0.964

Recall@5: 0.968

Recall@10: 1.000

MRR: 0.9463

5.2 Lexical Baseline (TF-IDF)

Recall@1: 0.856

Recall@3: 0.884

Recall@5: 0.912

Recall@10: 1.000

MRR: 0.8861

6. Analysis and Interpretation

The results show that:

The semantic model outperforms the lexical baseline across all metrics.

The largest improvement is observed in Recall@1 and MRR, indicating better top-rank accuracy.

Semantic embeddings handle paraphrases and vocabulary variation more effectively.

TF-IDF performs well when surface-form overlap exists, making it suitable as a fallback method.

The hybrid retrieval strategy improves system robustness by reducing no-match scenarios.

7. Confidence Calibration and Thresholding

Raw similarity scores are not directly interpretable as probabilities. Therefore, a sigmoid-based calibration is applied:

confidence = σ(a · (sim − b))

A global threshold of 0.75 is used:

If confidence ≥ 0.75 → Eligible

Otherwise → Not eligible

Fallback logic:

If the best semantic confidence is below the threshold, TF-IDF retrieval is triggered.

Results are labeled with their retrieval source (semantic or lexical) and an explanation.

8. API Behavior

The FastAPI endpoint:

GET /search?query=...&k=5

Returns:

Top-k candidates

Similarity score

Calibrated confidence

Eligibility flag

Retrieval source

Explanation

Health check:

GET /health

9. Reproducibility
9.1 Local Execution
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_make_synthetic_data.py
python scripts/02_build_index.py
python scripts/03_evaluate.py
python scripts/04_evaluate_tfidf.py

uvicorn app.main:app --host 0.0.0.0 --port 8000

9.2 Docker Execution
docker build -t semantic-matching .
docker run -p 8000:8000 semantic-matching

10. Conclusion

This project demonstrates a robust semantic matching pipeline for equipment eligibility classification. By combining dense semantic retrieval with lexical fallback, the system achieves high recall, strong top-rank accuracy, and practical interpretability.

The modular design allows for future extensions such as:

Cross-encoder reranking

Multilingual models

Multimodal matching using images

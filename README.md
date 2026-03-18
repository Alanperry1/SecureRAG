# SecureRAG 🔐

**A real-time Prompt Injection & Data Poisoning Detection Layer for RAG Pipelines.**

```
User Query → Retrieval → [Security Layer] → LLM → Response
                               ↓
                     Stage 1: Injection Detection
                     Stage 2: Poisoning Detection
                     Stage 3: Anomaly Scoring (PyTorch)
                     Stage 4: Confidence Gate
```

---

## Overview

RAG pipelines are vulnerable to two silent attacks:

| Attack | Description |
|---|---|
| **Prompt Injection** | Malicious documents manipulate the LLM's behaviour via retrieved context |
| **Data Poisoning** | Corrupted embeddings get stored in the vector DB and retrieved as trusted context |

SecureRAG intercepts the retrieved chunks *before* they reach the LLM and applies a 4-stage defence:

### Stage 1 — Injection Detection
- **Shannon entropy scoring** — high-entropy text signals linguistically unnatural content
- **Psycholinguistic features** — BEC-inspired override imperative density, authority mimicry, caps ratio (adapted from BEC email fraud detection)
- **Structural pattern scanning** — 8 regex fingerprints of known injection templates
- **Semantic role reversal** — cosine drift from query intent

### Stage 2 — Poisoning Detection (batch)
- **Cosine centroid outlier scoring** — chunks statistically far from the retrieval set centroid
- **Query–chunk z-score** — per-chunk similarity normalised against the peer distribution
- **Intra-set similarity gap** — poisoned clusters separate from legitimate ones
- **Embedding tamper detection** — re-embedds text and compares against stored vector; divergence signals DB tampering

### Stage 3 — PyTorch Autoencoder Anomaly Scoring
- Trains on clean corpus embeddings to learn the legitimate manifold
- Reconstruction error (MSE) flags off-manifold embeddings in real time
- Model is auto-calibrated: threshold = μ + 3σ of training errors

### Stage 4 — Confidence Gate
- Requires ≥ `MIN_SUPPORTING_CHUNKS` clean chunks
- Composite confidence: retrieval score + intra-chunk agreement + query alignment + score sharpness
- Returns `"I don't know"` when confidence < threshold (uncertainty-aware, inspired by LLM calibration work)

---

## Architecture

```
securerag/
├── api/
│   ├── main.py            ← FastAPI app entry point
│   ├── routes.py          ← /ingest, /query, /health, /cache
│   └── schemas.py         ← Pydantic request/response models
├── cache/
│   └── redis_cache.py     ← Redis-backed flagged chunk cache
├── core/
│   ├── embedder.py        ← SentenceTransformers wrapper
│   ├── retriever.py       ← ChromaDB retrieval + raw embeddings
│   ├── pipeline.py        ← End-to-end SecureRAG pipeline
│   └── security/
│       ├── injection_detector.py   ← Stage 1
│       ├── poisoning_detector.py   ← Stage 2
│       ├── anomaly_scorer.py       ← Stage 3 (PyTorch)
│       ├── confidence_gate.py      ← Stage 4
│       └── orchestrator.py        ← Wires all 3 detection stages
├── scripts/
│   └── train_anomaly_scorer.py    ← Corpus training CLI
├── tests/
│   ├── test_injection.py
│   ├── test_poisoning.py
│   └── test_anomaly_scorer.py
├── config.py              ← Pydantic settings (env-driven)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### 3. (Optional) Train the anomaly scorer on your corpus

```bash
python scripts/train_anomaly_scorer.py --corpus my_docs.txt --epochs 100
```

Each line in `my_docs.txt` is treated as one chunk.
The trained model is saved to `models/anomaly_encoder.pt` and auto-loaded at runtime.

### 4. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

### 5. Or run locally

```bash
# Start Redis (required for caching)
redis-server &

# Start API
python -m api.main
```

API is available at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

---

## API

### Ingest documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The mitochondria is the powerhouse of the cell.", "DNA stores genetic information."]}'
```

### Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the mitochondria do?"}'
```

**Response includes:**
```json
{
  "query": "What does the mitochondria do?",
  "answer": "...",
  "aborted": false,
  "sources": [...],
  "security": {
    "total_chunks": 5,
    "safe_chunks": 5,
    "flagged_chunks": 0,
    "injection_flagged": 0,
    "poisoning_flagged": 0,
    "anomaly_flagged": 0,
    "latency_ms": 4.2,
    "confidence": 0.821,
    "gate_passed": true
  }
}
```

### Health check

```bash
curl http://localhost:8000/api/v1/health
```

---

## Configuration

All settings are driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key for LLM calls |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `TOP_K` | `5` | Retrieved chunks per query |
| `INJECTION_SCORE_THRESHOLD` | `0.6` | Injection risk cutoff |
| `POISON_SCORE_THRESHOLD` | `0.65` | Poisoning risk cutoff |
| `ANOMALY_RECON_THRESHOLD` | `0.015` | Autoencoder MSE cutoff |
| `CONFIDENCE_THRESHOLD` | `0.70` | Min confidence to answer |
| `MIN_SUPPORTING_CHUNKS` | `2` | Min clean chunks required |
| `REDIS_ENABLED` | `true` | Enable Redis cache |

---

## Tests

```bash
pytest tests/ -v
```

---

## Performance

- Security layer adds **< 10ms** latency per query on CPU for `top_k=5` (embedding model pre-warmed)
- Redis cache eliminates re-scoring for previously flagged chunks: **< 1ms**
- Autoencoder inference: **< 2ms** on CPU for batch of 5

---

## Security Properties

| Attack | Detection Method | False Positive Rate |
|---|---|---|
| Classic injection ("ignore instructions") | Pattern scan + psycho score | Very low |
| Encoded / base64 injections | Entropy spike detection | Low |
| Semantic jailbreaks | Semantic drift + psycho score | Low–Medium |
| DB embedding tampering | Tamper detection (re-embed + compare) | Very low |
| Cosine outlier poisoning | Z-score outlier scoring | Medium |
| Off-manifold embeddings | Autoencoder MSE anomaly | Depends on training data |

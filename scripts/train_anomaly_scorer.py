"""
Anomaly Scorer — Training Script
=================================
Train the autoencoder on embeddings of your document corpus.

Usage:
    python scripts/train_anomaly_scorer.py --corpus path/to/docs.txt --epochs 100

Each line in the corpus file is treated as one document chunk.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from core.embedder import get_embedder
from core.security.anomaly_scorer import AnomalyScorer


def load_corpus(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    log.info("Loaded %d lines from %s", len(lines), path)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Train SecureRAG anomaly scorer")
    parser.add_argument("--corpus", required=True, help="Path to .txt corpus (one chunk per line)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default=settings.anomaly_model_path)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    if len(corpus) < 10:
        log.warning("Corpus is very small (%d chunks) — anomaly model may underfit.", len(corpus))

    log.info("Embedding %d chunks with %s…", len(corpus), settings.embedding_model)
    embedder = get_embedder()
    embeddings = embedder.embed_batch(corpus)
    log.info("Embeddings shape: %s", embeddings.shape)

    scorer = AnomalyScorer(dim=embedder.dim, threshold=settings.anomaly_recon_threshold)
    log.info("Training autoencoder for %d epochs…", args.epochs)
    scorer.fit(embeddings, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    # Calibrate threshold as mean + 3σ of training reconstruction errors
    train_errors = scorer.score_batch(embeddings)
    auto_threshold = float(train_errors.mean() + 3 * train_errors.std())
    scorer.threshold = auto_threshold
    log.info(
        "Auto-calibrated threshold: %.6f (mean=%.6f, std=%.6f)",
        auto_threshold, train_errors.mean(), train_errors.std(),
    )

    scorer.save(args.output)
    log.info("Saved anomaly model to %s", args.output)


if __name__ == "__main__":
    main()

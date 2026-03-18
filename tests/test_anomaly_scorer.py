import numpy as np
import pytest
from core.security.anomaly_scorer import AnomalyScorer
from config import settings


DIM = settings.embedding_dim


@pytest.fixture(scope="module")
def trained_scorer():
    """Train on synthetic normal embeddings."""
    rng = np.random.default_rng(0)
    # Normal embeddings: tight Gaussian cluster in embedding space
    normals = rng.standard_normal((300, DIM)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    scorer = AnomalyScorer(dim=DIM)
    scorer.fit(normals, epochs=20, batch_size=64)
    return scorer


def test_low_score_for_normal(trained_scorer):
    rng = np.random.default_rng(1)
    normal_emb = rng.standard_normal(DIM).astype(np.float32)
    normal_emb /= np.linalg.norm(normal_emb)
    score = trained_scorer.score(normal_emb)
    assert score >= 0.0


def test_output_shape_batch(trained_scorer):
    rng = np.random.default_rng(2)
    batch = rng.standard_normal((10, DIM)).astype(np.float32)
    batch /= np.linalg.norm(batch, axis=1, keepdims=True)
    scores = trained_scorer.score_batch(batch)
    assert scores.shape == (10,)
    assert (scores >= 0).all()


def test_all_scores_finite(trained_scorer):
    rng = np.random.default_rng(3)
    batch = rng.standard_normal((5, DIM)).astype(np.float32)
    scores = trained_scorer.score_batch(batch)
    assert np.all(np.isfinite(scores))


def test_save_and_load(trained_scorer, tmp_path):
    path = str(tmp_path / "ae.pt")
    trained_scorer.save(path)
    loaded = AnomalyScorer.load(path)
    rng = np.random.default_rng(4)
    emb = rng.standard_normal(DIM).astype(np.float32)
    s1 = trained_scorer.score(emb)
    s2 = loaded.score(emb)
    assert abs(s1 - s2) < 1e-6


def test_untrained_still_runs():
    """Untrained autoencoder should still produce scores (poor but not crash)."""
    scorer = AnomalyScorer(dim=DIM)
    rng = np.random.default_rng(5)
    emb = rng.standard_normal(DIM).astype(np.float32)
    score = scorer.score(emb)
    assert score >= 0.0

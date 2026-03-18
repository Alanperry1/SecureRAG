"""
PyTorch Anomaly Scorer — Autoencoder-based Embedding Anomaly Detection
=======================================================================
An autoencoder trained on normal document embeddings learns a compact
latent manifold.  Chunks with high reconstruction error (MSE) sit off
that manifold and are flagged as anomalous.

Usage
-----
Train:
    scorer = AnomalyScorer(dim=384)
    scorer.fit(normal_embeddings_np_array)
    scorer.save("models/anomaly_encoder.pt")

Inference:
    scorer = AnomalyScorer.load("models/anomaly_encoder.pt")
    score = scorer.score(chunk_embedding)   # float in [0, ∞)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import settings


class _Autoencoder(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AnomalyScorer:
    """Autoencoder-based anomaly scorer for embedding vectors."""

    def __init__(
        self,
        dim: int = settings.embedding_dim,
        bottleneck: int = 64,
        threshold: float = settings.anomaly_recon_threshold,
        device: str = "cpu",
    ) -> None:
        self.dim = dim
        self.threshold = threshold
        self.device = torch.device(device)
        self._model = _Autoencoder(dim, bottleneck).to(self.device)
        self._trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
    ) -> "AnomalyScorer":
        """Train on *clean* embeddings (shape: [N, dim])."""
        tensor = torch.tensor(embeddings, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                out = self._model(batch)
                loss = loss_fn(out, batch)
                optim.zero_grad()
                loss.backward()
                optim.step()

        self._trained = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, embedding: np.ndarray) -> float:
        """Return MSE reconstruction error.  Higher = more anomalous."""
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            recon = self._model(x)
            return float(nn.functional.mse_loss(recon, x).item())

    def score_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Return shape (N,) array of per-chunk MSE reconstruction errors."""
        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            recon = self._model(x)
            per_sample = ((recon - x) ** 2).mean(dim=1)
            return per_sample.cpu().numpy()

    def is_anomalous(self, embedding: np.ndarray) -> bool:
        return self.score(embedding) > self.threshold

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = settings.anomaly_model_path) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "dim": self.dim,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str = settings.anomaly_model_path) -> "AnomalyScorer":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        scorer = cls(dim=ckpt["dim"], threshold=ckpt["threshold"])
        scorer._model.load_state_dict(ckpt["state_dict"])
        scorer._trained = True
        return scorer

    @classmethod
    def load_or_init(cls, path: str = settings.anomaly_model_path) -> "AnomalyScorer":
        """Load saved model if it exists, otherwise return untrained instance."""
        if os.path.exists(path):
            return cls.load(path)
        return cls()

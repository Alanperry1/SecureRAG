"""
SecureRAG — Global Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── Vector Store ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "securerag"
    top_k: int = 5

    # ── Security Thresholds ───────────────────────────────────────────────────
    # Injection detection
    injection_entropy_threshold: float = 4.5   # bits — high entropy = suspicious
    injection_score_threshold: float = 0.6     # 0–1 composite score

    # Poisoning / cosine-outlier detection
    cosine_outlier_z_threshold: float = 2.0    # z-score below mean = outlier
    poison_score_threshold: float = 0.65

    # Anomaly autoencoder
    anomaly_recon_threshold: float = 0.015     # MSE reconstruction error
    anomaly_model_path: str = "./models/anomaly_encoder.pt"

    # Confidence gating
    confidence_threshold: float = 0.70         # must exceed to pass gate
    min_supporting_chunks: int = 2             # need ≥ N chunks supporting answer

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600              # cache flagged chunks for 1 h
    redis_enabled: bool = True

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

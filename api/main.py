"""
SecureRAG — FastAPI Application Entry Point
"""
import logging
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)

app = FastAPI(
    title="SecureRAG",
    description=(
        "Prompt Injection & Data Poisoning Detection Layer for RAG Pipelines.\n\n"
        "**Security Stages:**\n"
        "1. Injection detection (entropy + psycholinguistic + regex)\n"
        "2. Poisoning detection (cosine outlier z-scoring)\n"
        "3. Autoencoder anomaly scoring (PyTorch)\n"
        "4. Confidence gating (calibrated uncertainty)\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )

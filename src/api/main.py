"""FastAPI application for Social-AI-Detector.

Startup loads all detectors once; routes are non-blocking via run_in_executor
because Gemini embedding and GPU inference are synchronous operations.

Run from project root:
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models.ensemble import EnsembleDetector
from .models.knn import KNNDetector
from .models.llm import LlamaDetector
from .schemas import (
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    NeighborDetail,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the server can be run from anywhere)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "core"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

INDEX_PATH = str(PROCESSED_DIR / "corpus_balanced.index")
TRAIN_INDICES_PATH = str(PROCESSED_DIR / "train_indices_balanced.npy")
CORPUS_PATH = str(PROCESSED_DIR / "corpus.jsonl")
LLAMA_ADAPTER_PATH = str(MODELS_DIR / "llama_custom")
LLAMA_MLX_PATH = str(MODELS_DIR / "llama_mlx")

KNN_K = int(os.getenv("KNN_K", "10"))
ENSEMBLE_ALPHA = float(os.getenv("ENSEMBLE_ALPHA", "0.5"))


# ---------------------------------------------------------------------------
# Lifespan: load all detectors at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")

    knn = KNNDetector(
        index_path=INDEX_PATH,
        train_indices_path=TRAIN_INDICES_PATH,
        corpus_path=CORPUS_PATH,
        gemini_api_key=api_key,
        k=KNN_K,
    )
    llm = LlamaDetector(adapter_path=LLAMA_ADAPTER_PATH, mlx_path=LLAMA_MLX_PATH)
    ensemble = EnsembleDetector(knn=knn, llm=llm, alpha=ENSEMBLE_ALPHA)

    app.state.detectors = {
        knn.name: knn,
        llm.name: llm,
        ensemble.name: ensemble,
    }

    yield
    # No explicit cleanup needed; GC handles FAISS index and model weights.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Social AI Detector",
    description="Classifies social media text as AI-generated or human-written.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health(request: Request):
    detectors = request.app.state.detectors
    knn = detectors.get("knn")
    return HealthResponse(
        status="ok",
        faiss_vectors=knn._index.ntotal if knn else 0,
        corpus_size=len(knn._corpus) if knn else 0,
        detectors={name: d.is_available() for name, d in detectors.items()},
    )


@app.get("/api/models", response_model=ModelsResponse)
async def list_models(request: Request):
    detectors = request.app.state.detectors
    return ModelsResponse(models=[
        ModelInfo(
            name=d.name,
            description=d.description,
            available=d.is_available(),
        )
        for d in detectors.values()
    ])


@app.post("/api/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    detectors = request.app.state.detectors

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text field is empty")

    detector = detectors.get(body.model)
    if detector is None:
        valid = list(detectors.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model}'. Valid options: {valid}",
        )
    if not detector.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"Model '{body.model}' is not available (check GPU / dependencies).",
        )

    # Run synchronous predict() in thread pool so we don't block the event loop.
    # For the ensemble, pass through any per-request alpha override.
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()

    if body.model == "ensemble" and body.alpha is not None:
        predict_fn = partial(detector.predict, alpha=body.alpha)
    else:
        predict_fn = detector.predict

    results = await loop.run_in_executor(None, predict_fn, [text])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    r = results[0]
    return PredictResponse(
        prediction=r["prediction"],
        confidence=r["confidence"],
        model_used=detector.name,
        neighbors=[
            NeighborDetail(
                text_snippet=n["text_snippet"],
                label=n["label"],
                similarity=n["similarity"],
            )
            for n in r.get("neighbors", [])
        ],
        processing_time_ms=round(elapsed_ms, 1),
        knn_confidence=r.get("knn_confidence"),
        llm_confidence=r.get("llm_confidence"),
        alpha_used=r.get("alpha_used"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)

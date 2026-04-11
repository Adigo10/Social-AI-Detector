"""Pydantic request/response schemas for the Social-AI-Detector API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str
    model: str = "ensemble"
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ensemble blend weight for KNN (0=LLM-only, 1=KNN-only). "
                    "Only used when model='ensemble'. Falls back to server default if omitted.",
    )


class NeighborDetail(BaseModel):
    text_snippet: str   # first 200 chars
    label: str          # "ai" or "human"
    similarity: float   # cosine similarity from FAISS


class PredictResponse(BaseModel):
    prediction: str             # "ai" or "human"
    confidence: float           # P(AI), always in [0, 1]
    model_used: str
    neighbors: List[NeighborDetail]
    processing_time_ms: float
    knn_confidence: Optional[float] = None   # P(AI) from KNN alone
    llm_confidence: Optional[float] = None   # P(AI) from LLM alone
    alpha_used: Optional[float] = None       # actual α applied in ensemble


class ModelInfo(BaseModel):
    name: str
    description: str
    available: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    faiss_vectors: int
    corpus_size: int
    detectors: Dict[str, bool]

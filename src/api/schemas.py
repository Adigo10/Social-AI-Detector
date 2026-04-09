"""Pydantic request/response schemas for the Social-AI-Detector API."""

from typing import Dict, List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str
    model: str = "ensemble"


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

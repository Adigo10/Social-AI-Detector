"""Inference module for AI-generated text detection.

This module contains runtime components used by the backend for making predictions:
- retrieval: FAISS-based KNN search and classification
"""

from .retrieval import FAISSRetriever, RetrievalResult, Neighbour

__all__ = ["FAISSRetriever", "RetrievalResult", "Neighbour"]

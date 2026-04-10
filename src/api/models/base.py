"""Abstract base class that every detector must implement."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseDetector(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as the registry key and in API responses."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description returned by GET /api/models."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True only when all required resources are loaded and ready."""

    @abstractmethod
    def predict(
        self,
        texts: List[str],
        neighbors: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of texts.

        Args:
            texts: Raw text strings to classify.
            neighbors: Optional pre-computed FAISS neighbors, one list per text.
                Each neighbor dict has keys: text_snippet, label, similarity.
                When provided (by EnsembleDetector), the LLM reuses these for
                RAG context so FAISS is not queried a second time.

        Returns:
            List of dicts, one per input text::

                {
                    "prediction": "ai" | "human",
                    "confidence": float,   # P(AI) in [0, 1]
                    "neighbors": [...],    # list of neighbor dicts; empty if unsupported
                }

            len(result) == len(texts) always.
            Implementations must NOT raise on individual text failures —
            return {"prediction": "human", "confidence": 0.5, "neighbors": []}
            as a safe fallback.
        """

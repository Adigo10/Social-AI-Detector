"""EnsembleDetector: weighted combination of KNN and LLM predictions.

Architecture (from KNN doubts PDF):
- KNN (fast path): FAISS search → majority-vote confidence + neighbor list
- LLM (reasoning path): receives KNN neighbors as RAG context → token-logit confidence
- Ensemble: α·knn_conf + (1-α)·llm_conf

The same FAISS results from step 1 power both the LLM RAG context and the
ensemble KNN signal — no second FAISS query.

Degrades gracefully to KNN-only when the LLM is unavailable (no GPU, missing deps).
"""

from typing import Any, Dict, List, Optional

from .base import BaseDetector
from .knn import KNNDetector
from .llm import LlamaDetector


class EnsembleDetector(BaseDetector):

    def __init__(
        self,
        knn: KNNDetector,
        llm: LlamaDetector,
        alpha: float = 0.5,
    ):
        self._knn = knn
        self._llm = llm
        self._alpha = alpha

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def description(self) -> str:
        status = "KNN+LLM" if self._llm.is_available() else "KNN-only (LLM unavailable)"
        return f"Weighted ensemble ({status})"

    def is_available(self) -> bool:
        return self._knn.is_available()

    def predict(
        self,
        texts: List[str],
        neighbors: Optional[List[List[Dict[str, Any]]]] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        # Per-request alpha overrides the server default; fall back to self._alpha.
        a = self._alpha if alpha is None else float(alpha)

        # Step 1: KNN — embeds texts, queries FAISS, returns neighbors + confidence
        knn_results = self._knn.predict(texts)

        if not self._llm.is_available():
            # KNN-only fallback — annotate with alpha info for transparency
            for r in knn_results:
                r["knn_confidence"] = r["confidence"]
                r["llm_confidence"] = None
                r["alpha_used"] = 1.0  # effectively KNN-only
            return knn_results

        # Step 2: LLM — reuse KNN neighbors for RAG context (no second FAISS query)
        llm_neighbors = [r["neighbors"] for r in knn_results]
        llm_results = self._llm.predict(texts, neighbors=llm_neighbors)

        # Step 3: Weighted combination
        blended = []
        for knn_r, llm_r in zip(knn_results, llm_results):
            conf = a * knn_r["confidence"] + (1 - a) * llm_r["confidence"]
            blended.append({
                "prediction": "ai" if conf >= 0.5 else "human",
                "confidence": round(conf, 4),
                "neighbors": knn_r["neighbors"],  # KNN neighbors shown in UI
                "knn_confidence": round(knn_r["confidence"], 4),
                "llm_confidence": round(llm_r["confidence"], 4),
                "alpha_used": round(a, 4),
            })
        return blended

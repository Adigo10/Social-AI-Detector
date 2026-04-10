"""KNN detector: embed text via Gemini → FAISS search → majority-vote confidence.

Logic ported from src/eval/raid_eval.py:make_knn_predictor() with two additions:
  1. Structured as a class so resources (index, corpus, client) are loaded once.
  2. Returns neighbor details (text_snippet, label, similarity) for UI explainability.
"""

import json
import time
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from .base import BaseDetector

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIMENSIONS = 768
EMBED_BATCH_SIZE = 100
EMBED_SLEEP = 0.3
EMBED_RETRY_WAIT = 60
EMBED_MAX_RETRIES = 3
SNIPPET_MAX_CHARS = 200


class KNNDetector(BaseDetector):
    """K-Nearest Neighbors majority-vote classifier over a FAISS train index."""

    def __init__(
        self,
        index_path: str,
        train_indices_path: str,
        corpus_path: str,
        gemini_api_key: str,
        k: int = 10,
    ):
        from google import genai

        print("KNNDetector: loading FAISS index...")
        self._index = faiss.read_index(index_path)
        print(f"  Index size: {self._index.ntotal} vectors")

        print("KNNDetector: loading train indices mapping...")
        self._train_indices = np.load(train_indices_path)

        print("KNNDetector: loading corpus...")
        self._corpus: List[Dict] = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line))
        print(f"  Corpus size: {len(self._corpus)}")

        self._client = genai.Client(api_key=gemini_api_key)
        self._k = k
        self._available = True

    @property
    def name(self) -> str:
        return "knn"

    @property
    def description(self) -> str:
        return f"KNN majority vote on Gemini embeddings (k={self._k} train neighbors)"

    def is_available(self) -> bool:
        return self._available

    def _embed_batch(self, batch_texts: List[str]) -> Optional[List[List[float]]]:
        """Embed a batch with retry logic. Returns None on permanent failure."""
        from google.genai import types

        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                response = self._client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type="CLASSIFICATION",
                        output_dimensionality=EMBED_DIMENSIONS,
                    ),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                print(f"  KNN embed error (attempt {attempt}/{EMBED_MAX_RETRIES}): {e}")
                if attempt < EMBED_MAX_RETRIES:
                    time.sleep(EMBED_RETRY_WAIT)
        return None

    def predict(
        self,
        texts: List[str],
        neighbors: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        total = len(texts)

        # --- Embed ---
        embeddings = np.zeros((total, EMBED_DIMENSIONS), dtype=np.float32)
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[i: i + EMBED_BATCH_SIZE]
            result = self._embed_batch(batch)
            if result is None:
                # Return safe fallback for the whole batch on fatal embed failure
                return [
                    {"prediction": "human", "confidence": 0.5, "neighbors": []}
                    for _ in texts
                ]
            embeddings[i: i + len(batch)] = result
            if i > 0:
                time.sleep(EMBED_SLEEP)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # --- KNN search and majority vote ---
        results = []
        for i in range(total):
            # Use [i:i+1] slice — guaranteed C-contiguous for FAISS
            query = embeddings[i: i + 1]
            scores, idx_result = self._index.search(query, self._k)

            ai_votes = 0
            total_votes = 0
            neighbor_list = []

            for j, local_idx in enumerate(idx_result[0]):
                if local_idx < 0 or local_idx >= len(self._train_indices):
                    continue
                corpus_idx = self._train_indices[local_idx]
                record = self._corpus[corpus_idx]
                label = record["label"]
                similarity = float(scores[0][j])

                if label == "ai":
                    ai_votes += 1
                total_votes += 1

                neighbor_list.append({
                    "text_snippet": record["text"][:SNIPPET_MAX_CHARS],
                    "label": label,
                    "similarity": round(similarity, 4),
                })

            confidence = ai_votes / total_votes if total_votes > 0 else 0.5
            # Tie-break → "ai" (matches knn_baseline.py convention)
            prediction = "ai" if ai_votes >= (total_votes - ai_votes) else "human"

            results.append({
                "prediction": prediction,
                "confidence": confidence,
                "neighbors": neighbor_list,
            })

        return results

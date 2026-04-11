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


def compute_weighted_knn_result(
    neighbors: List[Dict[str, Any]],
) -> tuple[str, float]:
    """Return cosine-weighted KNN label and P(ai)."""
    ai_weight = 0.0
    human_weight = 0.0
    ai_count = 0
    human_count = 0

    for neighbor in neighbors:
        label = neighbor["label"]
        similarity = max(0.0, float(neighbor["similarity"]))
        if label == "ai":
            ai_weight += similarity
            ai_count += 1
        else:
            human_weight += similarity
            human_count += 1

    total_weight = ai_weight + human_weight
    if total_weight > 0:
        ai_confidence = ai_weight / total_weight
    else:
        total_count = ai_count + human_count
        ai_confidence = (ai_count / total_count) if total_count > 0 else 0.5

    ai_confidence = round(ai_confidence, 4)
    prediction = "ai" if ai_confidence >= 0.5 else "human"
    return prediction, ai_confidence


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
        print(f"  Index path: {index_path}")
        self._index = faiss.read_index(index_path)
        print(f"  Index size: {self._index.ntotal} vectors")

        print("KNNDetector: loading train indices mapping...")
        print(f"  Train indices path: {train_indices_path}")
        self._train_indices = np.load(train_indices_path)

        print("KNNDetector: loading corpus...")
        print(f"  Corpus path: {corpus_path}")
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
        from typing import Set
        total = len(texts)

        # --- Embed ---
        embeddings = np.zeros((total, EMBED_DIMENSIONS), dtype=np.float32)
        failed_indices: Set[int] = set()
        for i in range(0, total, EMBED_BATCH_SIZE):
            if i > 0:
                time.sleep(EMBED_SLEEP)  # rate-limit: sleep BEFORE each batch after the first
            batch = texts[i: i + EMBED_BATCH_SIZE]
            result = self._embed_batch(batch)
            if result is None:
                # Track failed indices; keep going so other batches still succeed
                print(f"  KNN: permanent embed failure for batch at index {i}; "
                      f"fallback for {len(batch)} texts.")
                failed_indices.update(range(i, i + len(batch)))
            else:
                embeddings[i: i + len(batch)] = result

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # --- KNN search and majority vote ---
        results = []
        for i in range(total):
            if i in failed_indices:
                results.append({
                    "prediction": "human",
                    "confidence": 0.5,
                    "neighbors": [],
                    "knn_confidence": 0.5,
                    "llm_confidence": None,
                    "alpha_used": 1.0,
                })
                continue

            # Use [i:i+1] slice — guaranteed C-contiguous for FAISS
            query = embeddings[i: i + 1]
            scores, idx_result = self._index.search(query, self._k)

            neighbor_list = []

            for j, local_idx in enumerate(idx_result[0]):
                if local_idx < 0 or local_idx >= len(self._train_indices):
                    continue
                corpus_idx = self._train_indices[local_idx]
                record = self._corpus[corpus_idx]
                label = record["label"]
                similarity = float(scores[0][j])

                neighbor_list.append({
                    "full_text": record["text"],
                    "text_snippet": record["text"][:SNIPPET_MAX_CHARS],
                    "label": label,
                    "similarity": similarity,
                })

            if not neighbor_list:
                # No valid FAISS results — safe default
                confidence, prediction = 0.5, "human"
            else:
                prediction, confidence = compute_weighted_knn_result(neighbor_list)

            display_neighbors = [
                {
                    "full_text": n["full_text"],
                    "text_snippet": n["text_snippet"],
                    "label": n["label"],
                    "similarity": round(float(n["similarity"]), 4),
                }
                for n in neighbor_list
            ]
            results.append({
                "prediction": prediction,
                "confidence": confidence,
                "neighbors": display_neighbors,
                "knn_confidence": confidence,
                "llm_confidence": None,
                "alpha_used": 1.0,
            })

        return results

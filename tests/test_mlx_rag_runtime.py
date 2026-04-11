from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


knn_module = importlib.import_module("src.api.models.knn")
llm_module = importlib.import_module("src.api.models.llm")


def test_weighted_knn_prefers_stronger_similarity_even_if_outnumbered():
    neighbors = [
        {"label": "human", "similarity": 0.10},
        {"label": "human", "similarity": 0.10},
        {"label": "ai", "similarity": 0.95},
    ]

    prediction, confidence = knn_module.compute_weighted_knn_result(neighbors)

    assert prediction == "ai"
    assert confidence == 0.8261


def test_build_rag_pairs_prefers_full_text_over_ui_snippet():
    neighbors = [
        {
            "full_text": "full retrieved text that should reach the prompt",
            "text_snippet": "truncated ui snippet",
            "label": "human",
        }
    ]

    pairs = llm_module.build_rag_pairs_from_neighbors(neighbors)

    assert pairs == [("full retrieved text that should reach the prompt", "human")]

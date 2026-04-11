#!/usr/bin/env python3
"""Run local ensemble inference with KNN retrieval and MLX LLM scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.models.ensemble import EnsembleDetector
from src.api.models.knn import KNNDetector
from src.api.models.llm import LlamaDetector


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "core"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_PATH = str(PROCESSED_DIR / "corpus_balanced.index")
TRAIN_INDICES_PATH = str(PROCESSED_DIR / "train_indices_balanced.npy")
CORPUS_PATH = str(PROCESSED_DIR / "corpus.jsonl")
LLAMA_ADAPTER_PATH = str(MODELS_DIR / "llama_custom")
LLAMA_MLX_PATH = str(MODELS_DIR / "llama_mlx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local ensemble inference with MLX-backed LLM."
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Raw social media post text.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of FAISS neighbors for KNN retrieval.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend weight for KNN confidence (0=LLM only, 1=KNN only).",
    )
    parser.add_argument(
        "--show-neighbors",
        type=int,
        default=5,
        help="How many retrieved neighbors to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    return parser.parse_args()


def require_gemini_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your .env file.")
    return api_key


def build_ensemble(k: int, alpha: float) -> EnsembleDetector:
    api_key = require_gemini_api_key()
    knn = KNNDetector(
        index_path=INDEX_PATH,
        train_indices_path=TRAIN_INDICES_PATH,
        corpus_path=CORPUS_PATH,
        gemini_api_key=api_key,
        k=k,
    )
    llm = LlamaDetector(adapter_path=LLAMA_ADAPTER_PATH, mlx_path=LLAMA_MLX_PATH)
    if not llm.is_available() or llm._device != "mlx":
        raise RuntimeError(
            "MLX LLM path unavailable. Check mlx_lm install and models/llama_mlx."
        )
    return EnsembleDetector(knn=knn, llm=llm, alpha=alpha)


def run_inference(text: str, k: int, alpha: float) -> dict:
    ensemble = build_ensemble(k=k, alpha=alpha)
    result = ensemble.predict([text], alpha=alpha)[0]
    return {
        "text": text,
        "ensemble_mlx": result,
    }


def print_human(result: dict, show_neighbors: int) -> None:
    ensemble_result = result["ensemble_mlx"]
    print("Ensemble prediction")
    print(f"- Prediction: {ensemble_result['prediction']}")
    print(f"- P(ai): {ensemble_result['confidence']:.4f}")
    print(f"- KNN P(ai): {ensemble_result['knn_confidence']:.4f}")
    print(f"- LLM P(ai): {ensemble_result['llm_confidence']:.4f}")
    print(f"- Alpha: {ensemble_result['alpha_used']:.4f}")

    print("Top neighbors")
    for idx, neighbor in enumerate(ensemble_result["neighbors"][:show_neighbors], start=1):
        print(
            f"- [{idx}] {neighbor['label']} | sim={neighbor['similarity']:.4f} | "
            f"{neighbor['text_snippet']}"
        )


def main() -> None:
    args = parse_args()
    result = run_inference(text=args.text, k=args.k, alpha=args.alpha)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print_human(result, show_neighbors=args.show_neighbors)


if __name__ == "__main__":
    main()

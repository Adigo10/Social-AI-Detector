"""RAID adversarial evaluation harness for AI-generated text detection.

Evaluates any model against the RAID adversarial test set (fully unseen
during training). Accepts a prediction function via a simple callable
interface and reports TPR@1%FPR (RAID benchmark standard), AUROC,
Macro-F1, and Accuracy — both overall and broken down by source_model.

Usage as CLI:
    python src/eval/raid_eval.py --model knn --k 10
    python src/eval/raid_eval.py --model dummy

Usage as library:
    from src.eval.raid_eval import evaluate_raid
    results = evaluate_raid(my_predict_fn)
"""

import argparse
import json
import os
import sys
import time

import faiss
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAID_EVAL_PATH = os.path.join(PROCESSED_DIR, "raid_eval.jsonl")
RAID_RESULTS_PATH = os.path.join(PROCESSED_DIR, "raid_results.json")
INDEX_PATH = os.path.join(PROCESSED_DIR, "corpus.index")
TRAIN_INDICES_PATH = os.path.join(PROCESSED_DIR, "train_indices.npy")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")

EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_DIMENSIONS = 768
EMBED_BATCH_SIZE = 100
EMBED_SLEEP = 0.3
EMBED_RETRY_WAIT = 60
EMBED_MAX_RETRIES = 3

PROGRESS_INTERVAL = 1_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raid_data():
    """Load RAID evaluation records from raid_eval.jsonl."""
    records = []
    with open(RAID_EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} RAID evaluation records")
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_scores, y_pred):
    """Compute TPR@1%FPR, AUROC, Macro-F1, and Accuracy.

    Args:
        y_true: array of int (1=ai, 0=human)
        y_scores: array of float (confidence that label is ai)
        y_pred: array of int (1=ai, 0=human)

    Returns:
        dict with metric values
    """
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)

    # Need both classes for meaningful metrics
    if n_pos == 0 or n_neg == 0:
        return {
            "n_records": len(y_true),
            "n_ai": n_pos,
            "n_human": n_neg,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": None,
            "auroc": None,
            "tpr_at_1fpr": None,
        }

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auroc = roc_auc_score(y_true, y_scores)

    # TPR @ 1% FPR (RAID benchmark standard)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tpr_at_1fpr = float(np.interp(0.01, fpr, tpr))

    return {
        "n_records": len(y_true),
        "n_ai": n_pos,
        "n_human": n_neg,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "auroc": round(auroc, 4),
        "tpr_at_1fpr": round(tpr_at_1fpr, 4),
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_raid(predict_fn, records=None):
    """Evaluate a model against the RAID adversarial test set.

    Args:
        predict_fn: callable that takes a list of texts and returns a list
            of dicts with {"prediction": "ai"/"human", "confidence": float}.
            confidence is the model's confidence that the text is AI-generated
            (higher = more likely AI).
        records: optional pre-loaded RAID records. If None, loads from disk.

    Returns:
        dict with "overall" metrics and "by_source_model" breakdown.
    """
    if records is None:
        records = load_raid_data()

    texts = [r["text"] for r in records]
    true_labels = [r["label"] for r in records]

    print(f"\nRunning predictions on {len(texts)} RAID texts...")
    start = time.time()
    predictions = predict_fn(texts)
    elapsed = time.time() - start
    print(f"Predictions complete in {elapsed:.1f}s")

    assert len(predictions) == len(texts), (
        f"Prediction count mismatch: got {len(predictions)}, expected {len(texts)}"
    )

    # Convert to arrays
    y_true = np.array([1 if l == "ai" else 0 for l in true_labels])
    y_pred = np.array([1 if p["prediction"] == "ai" else 0 for p in predictions])
    y_scores = np.array([p["confidence"] for p in predictions])

    # Overall metrics
    overall = compute_metrics(y_true, y_scores, y_pred)

    # Breakdown by source_model
    source_models = sorted(set(r.get("source_model", "unknown") for r in records))
    by_source = {}
    for model_name in source_models:
        mask = np.array([
            r.get("source_model", "unknown") == model_name for r in records
        ])
        if mask.sum() == 0:
            continue
        m = compute_metrics(y_true[mask], y_scores[mask], y_pred[mask])
        m["source_model"] = model_name
        by_source[model_name] = m

    results = {
        "overall": overall,
        "by_source_model": by_source,
    }
    return results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_results(results):
    """Print a formatted summary table of RAID evaluation results."""
    overall = results["overall"]
    by_source = results["by_source_model"]

    print("\n" + "=" * 75)
    print("RAID Adversarial Evaluation Results")
    print("=" * 75)

    # Overall
    print(f"\nOverall ({overall['n_records']} records: "
          f"{overall['n_ai']} AI, {overall['n_human']} human)")
    print(f"  TPR@1%FPR:  {_fmt(overall['tpr_at_1fpr'])}")
    print(f"  AUROC:      {_fmt(overall['auroc'])}")
    print(f"  Macro-F1:   {_fmt(overall['macro_f1'])}")
    print(f"  Accuracy:   {_fmt(overall['accuracy'])}")

    # By source model
    if by_source:
        print(f"\n{'Source Model':<20} {'N':>6} {'TPR@1%FPR':>10} "
              f"{'AUROC':>8} {'Macro-F1':>10} {'Acc':>8}")
        print("-" * 75)
        for name in sorted(by_source.keys()):
            m = by_source[name]
            print(
                f"{name:<20} {m['n_records']:>6} "
                f"{_fmt(m['tpr_at_1fpr']):>10} "
                f"{_fmt(m['auroc']):>8} "
                f"{_fmt(m['macro_f1']):>10} "
                f"{_fmt(m['accuracy']):>8}"
            )
    print("=" * 75)


def _fmt(value):
    """Format a metric value, handling None."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Built-in predictors
# ---------------------------------------------------------------------------

def dummy_predictor(texts):
    """Dummy predictor: randomly guesses with 50/50 confidence."""
    rng = np.random.default_rng(42)
    results = []
    for _ in texts:
        pred = rng.choice(["ai", "human"])
        conf = rng.uniform(0.3, 0.7)
        if pred == "human":
            conf = 1.0 - conf
        results.append({"prediction": pred, "confidence": conf})
    return results


def make_knn_predictor(k=10):
    """Build a KNN predictor that embeds RAID texts and queries the train index.

    Returns a predict_fn compatible with evaluate_raid().
    """
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Set it in .env file.")
        sys.exit(1)

    # Load train-only FAISS index and corpus
    print("Loading FAISS index and corpus for KNN...")
    index = faiss.read_index(INDEX_PATH)
    train_indices = np.load(TRAIN_INDICES_PATH)
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"  Index: {index.ntotal} vectors, Corpus: {len(corpus)} records")

    client = genai.Client(api_key=api_key)

    def _embed_batch(batch_texts):
        """Embed a batch with retry logic."""
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                response = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type="CLASSIFICATION",
                        output_dimensionality=EMBED_DIMENSIONS,
                    ),
                )
                return [e.values for e in response.embeddings]
            except Exception as e:
                print(f"  API error (attempt {attempt}/{EMBED_MAX_RETRIES}): {e}")
                if attempt < EMBED_MAX_RETRIES:
                    print(f"  Waiting {EMBED_RETRY_WAIT}s before retry...")
                    time.sleep(EMBED_RETRY_WAIT)
        return None

    def predict_fn(texts):
        total = len(texts)
        print(f"\nEmbedding {total} RAID texts via Gemini API...")

        # Generate embeddings in batches
        embeddings = np.zeros((total, EMBED_DIMENSIONS), dtype=np.float32)
        for i in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            batch_end = min(i + EMBED_BATCH_SIZE, total)
            result = _embed_batch(batch)
            if result is None:
                print(f"FATAL: Embedding failed at index {i}")
                sys.exit(1)
            embeddings[i:batch_end] = result

            if (batch_end) % PROGRESS_INTERVAL == 0 or batch_end == total:
                print(f"  [{batch_end}/{total}] {100*batch_end/total:.1f}%")
            time.sleep(EMBED_SLEEP)

        # L2-normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # Query index and majority vote
        print(f"Running KNN search (k={k})...")
        predictions = []
        for i in range(total):
            query = embeddings[i : i + 1]
            scores, indices_result = index.search(query, k)

            ai_votes = 0
            total_votes = 0
            for local_idx in indices_result[0]:
                if local_idx < 0 or local_idx >= len(train_indices):
                    continue
                corpus_idx = train_indices[local_idx]
                if corpus[corpus_idx]["label"] == "ai":
                    ai_votes += 1
                total_votes += 1

            ai_confidence = ai_votes / total_votes if total_votes > 0 else 0.5

            if ai_votes >= total_votes - ai_votes:
                pred = "ai"
            else:
                pred = "human"

            predictions.append({"prediction": pred, "confidence": ai_confidence})

            if (i + 1) % PROGRESS_INTERVAL == 0 or i == total - 1:
                print(f"  KNN [{i+1}/{total}] {100*(i+1)/total:.1f}%")

        return predictions

    return predict_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model against the RAID adversarial test set"
    )
    parser.add_argument(
        "--model", type=str, default="dummy",
        choices=["dummy", "knn"],
        help="Predictor to use (default: dummy)"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of neighbors for KNN predictor (default: 10)"
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save results to raid_results.json (default: True)"
    )
    parser.add_argument(
        "--no-save", action="store_false", dest="save",
        help="Do not save results"
    )
    args = parser.parse_args()

    # Select predictor
    if args.model == "dummy":
        print("Using dummy (random) predictor")
        predict_fn = dummy_predictor
    elif args.model == "knn":
        print(f"Using KNN predictor (k={args.k})")
        predict_fn = make_knn_predictor(k=args.k)

    # Run evaluation
    results = evaluate_raid(predict_fn)
    results["model"] = args.model
    if args.model == "knn":
        results["k"] = args.k

    # Print results
    print_results(results)

    # Save results
    if args.save:
        with open(RAID_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        size_mb = os.path.getsize(RAID_RESULTS_PATH) / (1024 * 1024)
        print(f"\nSaved: {RAID_RESULTS_PATH} ({size_mb:.3f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()

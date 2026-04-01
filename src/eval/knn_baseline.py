"""KNN majority-vote baseline using the train-only FAISS index.

For each val/test record, retrieves k nearest train neighbors via cosine
similarity (L2-normalized inner product), then predicts by majority vote.
Reports TPR@1%FPR, AUROC, Macro-F1, and Accuracy.
"""

import argparse
import json
import os

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
INDEX_PATH = os.path.join(PROCESSED_DIR, "corpus.index")
TRAIN_INDICES_PATH = os.path.join(PROCESSED_DIR, "train_indices.npy")
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings.npy")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
SPLITS_PATH = os.path.join(PROCESSED_DIR, "splits.json")
RESULTS_PATH = os.path.join(PROCESSED_DIR, "knn_results.json")

PROGRESS_INTERVAL = 5_000


def load_data():
    """Load corpus, splits, embeddings, FAISS index, and train index mapping."""
    print("=== Loading Data ===")

    print("Loading corpus...")
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"  Corpus size: {len(corpus)}")

    print("Loading splits...")
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  Embeddings shape: {embeddings.shape}")

    assert len(corpus) == embeddings.shape[0], (
        f"Size mismatch: corpus={len(corpus)}, embeddings={embeddings.shape[0]}"
    )

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    print(f"  Index size: {index.ntotal} vectors (train only)")

    print("Loading train indices mapping...")
    train_indices = np.load(TRAIN_INDICES_PATH)
    assert index.ntotal == len(train_indices), (
        f"Index/mapping mismatch: index has {index.ntotal} vectors "
        f"but mapping has {len(train_indices)} entries"
    )

    return corpus, splits, embeddings, index, train_indices


def evaluate_split(split_name, record_indices, corpus, embeddings,
                   index, train_indices, k):
    """Run KNN majority-vote classification on a single split.

    Returns dict with predictions, labels, confidence scores, and metrics.
    """
    count = len(record_indices)
    if count == 0:
        print(f"  No records for {split_name}, skipping.")
        return None

    print(f"\n=== Evaluating {split_name} split ({count} records, k={k}) ===")

    y_true = []       # 1 = ai, 0 = human
    y_pred = []       # 1 = ai, 0 = human
    y_scores = []     # proportion of "ai" votes (confidence)

    for progress, i in enumerate(record_indices):
        record = corpus[i]
        true_label = record["label"]
        y_true.append(1 if true_label == "ai" else 0)

        # L2-normalize query vector
        query = embeddings[i:i + 1].copy().astype(np.float32)
        faiss.normalize_L2(query)

        # Retrieve k neighbors from train-only index
        scores, indices_result = index.search(query, k)

        # Collect neighbor labels (map FAISS positions → corpus indices)
        ai_votes = 0
        total_votes = 0
        for local_idx in indices_result[0]:
            if local_idx < 0 or local_idx >= len(train_indices):
                continue
            corpus_idx = train_indices[local_idx]
            neighbor_label = corpus[corpus_idx]["label"]
            if neighbor_label == "ai":
                ai_votes += 1
            total_votes += 1

        # Confidence: proportion of ai votes
        ai_confidence = ai_votes / total_votes if total_votes > 0 else 0.5
        y_scores.append(ai_confidence)

        # Majority vote (tie → predict ai)
        if ai_votes >= total_votes - ai_votes:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if (progress + 1) % PROGRESS_INTERVAL == 0 or progress == count - 1:
            print(f"  [{progress+1}/{count}] {100*(progress+1)/count:.1f}%")

    # Compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auroc = roc_auc_score(y_true, y_scores)

    # TPR @ 1% FPR
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    tpr_at_1fpr = float(np.interp(0.01, fpr, tpr))

    metrics = {
        "split": split_name,
        "k": k,
        "n_records": count,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "auroc": round(auroc, 4),
        "tpr_at_1fpr": round(tpr_at_1fpr, 4),
    }

    return metrics


def print_summary(results):
    """Print a formatted summary table of evaluation results."""
    print("\n" + "=" * 65)
    print(f"{'Split':<8} {'k':>4} {'Acc':>8} {'Macro-F1':>10} {'AUROC':>8} {'TPR@1%FPR':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['split']:<8} {r['k']:>4} "
            f"{r['accuracy']:>8.4f} {r['macro_f1']:>10.4f} "
            f"{r['auroc']:>8.4f} {r['tpr_at_1fpr']:>10.4f}"
        )
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="KNN majority-vote baseline for AI text detection"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of nearest neighbors (default: 10)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to knn_results.json"
    )
    args = parser.parse_args()

    corpus, splits, embeddings, index, train_indices = load_data()

    # Group corpus indices by split
    split_indices = {"val": [], "test": []}
    for i, record in enumerate(corpus):
        split_name = splits.get(str(record["id"]))
        if split_name in split_indices:
            split_indices[split_name].append(i)

    for name, indices in split_indices.items():
        print(f"  {name}: {len(indices)} records")

    # Evaluate val and test splits
    results = []
    for split_name in ("val", "test"):
        metrics = evaluate_split(
            split_name, split_indices[split_name],
            corpus, embeddings, index, train_indices, args.k,
        )
        if metrics:
            results.append(metrics)

    # Print summary
    print_summary(results)

    # Optionally save results
    if args.save:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        size_mb = os.path.getsize(RESULTS_PATH) / (1024 * 1024)
        print(f"\nSaved: {RESULTS_PATH} ({size_mb:.3f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()

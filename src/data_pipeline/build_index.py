"""Step 4: Build FAISS index from train-split embeddings only.

This builds a train-only index to prevent data leakage: val/test embeddings
are excluded so retrieval-based features never expose held-out data.

Note: embeddings.npy contains raw (un-normalized) vectors from the Gemini API.
L2 normalization is applied here at index-build time so that inner product = cosine similarity.
"""

import json
import os

import faiss
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings.npy")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
SPLITS_PATH = os.path.join(PROCESSED_DIR, "splits.json")
INDEX_PATH = os.path.join(PROCESSED_DIR, "corpus.index")
TRAIN_INDICES_PATH = os.path.join(PROCESSED_DIR, "train_indices.npy")
DIMENSIONS = 768


def main():
    # Load splits to identify training indices
    print("Loading splits...")
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    print("Loading corpus to map IDs to splits...")
    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    train_indices = []
    for i, record in enumerate(corpus):
        if splits.get(str(record["id"])) == "train":
            train_indices.append(i)

    print(f"  Total corpus: {len(corpus)}, Train split: {len(train_indices)}")

    # Load embeddings and select train subset
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")

    train_embeddings = embeddings[train_indices].copy()

    # L2-normalize so inner product = cosine similarity
    print("L2-normalizing train vectors...")
    faiss.normalize_L2(train_embeddings)

    # Build flat inner product index (exact search)
    print(f"Building IndexFlatIP (dim={DIMENSIONS})...")
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(train_embeddings)
    print(f"  Index size: {index.ntotal} vectors (train only)")

    # Save index and train indices mapping (for downstream consumers)
    faiss.write_index(index, INDEX_PATH)
    np.save(TRAIN_INDICES_PATH, np.array(train_indices, dtype=np.int64))
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"  Saved: {INDEX_PATH} ({size_mb:.1f} MB)")
    print(f"  Saved: {TRAIN_INDICES_PATH} ({len(train_indices)} indices)")

    # Sanity check: query the first train vector
    print("\n--- Sanity Check ---")
    query = train_embeddings[0:1].copy()
    # Already normalized above, but normalize_L2 on a unit vector is a no-op
    scores, indices_result = index.search(query, 5)
    print("Top 5 neighbors of train vector 0:")
    for rank, (idx, score) in enumerate(zip(indices_result[0], scores[0])):
        print(f"  #{rank+1}: index={idx}, similarity={score:.6f}")

    if indices_result[0][0] == 0 and scores[0][0] > 0.99:
        print("  ✓ First result is itself with similarity ~1.0 (as expected)")
    else:
        print("  ⚠ WARNING: First result is not itself — check data alignment")


if __name__ == "__main__":
    main()

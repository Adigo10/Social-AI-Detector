"""Step 4: Build FAISS index from embeddings."""

import os

import numpy as np
import faiss

PROCESSED_DIR = os.path.join("data", "processed")
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(PROCESSED_DIR, "corpus.index")
DIMENSIONS = 768


def main():
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    n_vectors = embeddings.shape[0]

    # L2-normalize so inner product = cosine similarity
    print("L2-normalizing vectors...")
    faiss.normalize_L2(embeddings)

    # Build flat inner product index (exact search)
    print(f"Building IndexFlatIP (dim={DIMENSIONS})...")
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(embeddings)
    print(f"  Index size: {index.ntotal} vectors")

    # Save index
    faiss.write_index(index, INDEX_PATH)
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"  Saved: {INDEX_PATH} ({size_mb:.1f} MB)")

    # Sanity check: query the first vector
    print("\n--- Sanity Check ---")
    query = embeddings[0:1].copy()
    faiss.normalize_L2(query)
    scores, indices = index.search(query, 5)
    print("Top 5 neighbors of vector 0:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        print(f"  #{rank+1}: index={idx}, similarity={score:.6f}")

    if indices[0][0] == 0 and scores[0][0] > 0.99:
        print("  ✓ First result is itself with similarity ~1.0 (as expected)")
    else:
        print("  ⚠ WARNING: First result is not itself — check data alignment")


if __name__ == "__main__":
    main()

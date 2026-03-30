"""Step 3: Generate embeddings via Gemini API with checkpoint/resume."""

import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROCESSED_DIR = os.path.join("data", "processed")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "embeddings.npy")
CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, "embeddings_checkpoint.npz")

MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 768
BATCH_SIZE = 100
CHECKPOINT_INTERVAL = 10_000
SLEEP_BETWEEN_CALLS = 0.3
RETRY_WAIT = 60
MAX_RETRIES = 3


def load_texts():
    """Load all texts from corpus.jsonl in order."""
    texts = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def load_checkpoint():
    """Load checkpoint if it exists. Returns (embeddings_so_far, next_index)."""
    if os.path.exists(CHECKPOINT_PATH):
        data = np.load(CHECKPOINT_PATH)
        embeddings = data["embeddings"]
        next_index = int(data["next_index"])
        print(f"Resuming from checkpoint: {next_index} texts already embedded")
        return embeddings, next_index
    return None, 0


def save_checkpoint(embeddings, next_index):
    """Save checkpoint with current progress."""
    assert embeddings.shape[0] == next_index, (
        f"Checkpoint alignment error: {embeddings.shape[0]} embeddings but next_index={next_index}"
    )
    np.savez(CHECKPOINT_PATH, embeddings=embeddings, next_index=next_index)
    print(f"  Checkpoint saved at index {next_index}")


def embed_batch(client, texts):
    """Embed a batch of texts with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="CLASSIFICATION",
                    output_dimensionality=DIMENSIONS,
                ),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            print(f"  API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                print(f"  Waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)
    return None  # All retries failed


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Set it in .env file.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Loading corpus texts...")
    texts = load_texts()
    total = len(texts)
    print(f"Total texts: {total}")

    # Load or initialize embeddings
    existing_embeddings, start_index = load_checkpoint()
    if existing_embeddings is not None:
        embeddings = list(existing_embeddings)
    else:
        embeddings = []

    start_time = time.time()
    processed = start_index

    print(f"\nStarting embedding from index {start_index}...")
    for i in range(start_index, total, BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_end = min(i + BATCH_SIZE, total)

        result = embed_batch(client, batch_texts)
        if result is None:
            # All retries failed — save checkpoint and exit
            print(f"\nFATAL: API failed after {MAX_RETRIES} retries at index {i}")
            if embeddings:
                arr = np.array(embeddings, dtype=np.float32)
                save_checkpoint(arr, len(embeddings))
            print("Checkpoint saved. Rerun this script to resume.")
            sys.exit(1)

        embeddings.extend(result)
        processed = batch_end
        elapsed = time.time() - start_time
        rate = (processed - start_index) / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        print(
            f"  [{processed}/{total}] "
            f"{100*processed/total:.1f}% | "
            f"{rate:.0f} texts/s | "
            f"ETA: {eta/60:.1f} min"
        )

        # Checkpoint every CHECKPOINT_INTERVAL texts
        if processed % CHECKPOINT_INTERVAL == 0 and processed > start_index:
            arr = np.array(embeddings, dtype=np.float32)
            save_checkpoint(arr, processed)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save final output
    arr = np.array(embeddings, dtype=np.float32)
    assert arr.shape[0] == total, (
        f"Embedding count mismatch: expected {total}, got {arr.shape[0]}"
    )
    np.save(OUTPUT_PATH, arr)

    # Remove checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint file removed.")

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nEmbedding complete!")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  File: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

"""Step 3: Generate embeddings via Gemini API with checkpoint/resume.

Usage:
    python embed.py                          # defaults: corpus.jsonl -> core/embeddings.npy
    python embed.py --input PATH --output PATH [--checkpoint PATH]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

DEFAULT_INPUT = os.path.join(PROCESSED_DIR, "core", "corpus.jsonl")
DEFAULT_OUTPUT = os.path.join(PROCESSED_DIR, "core", "embeddings.npy")
DEFAULT_CHECKPOINT = os.path.join(PROCESSED_DIR, "core", "embeddings_checkpoint.npz")

MODEL = "gemini-embedding-2-preview"
DIMENSIONS = 768
BATCH_SIZE = 100
CHECKPOINT_INTERVAL = 10_000
SLEEP_BETWEEN_CALLS = 0.3
RETRY_WAIT = 60
MAX_RETRIES = 3


def load_texts(input_path):
    """Load all texts from a .jsonl file in order."""
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists. Returns (embeddings_so_far, next_index)."""
    if os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path)
        embeddings = data["embeddings"]
        next_index = int(data["next_index"])
        print(f"Resuming from checkpoint: {next_index} texts already embedded")
        return embeddings, next_index
    return None, 0


def save_checkpoint(checkpoint_path, embeddings, next_index):
    """Save checkpoint with current progress."""
    assert embeddings.shape[0] == next_index, (
        f"Checkpoint alignment error: {embeddings.shape[0]} embeddings but next_index={next_index}"
    )
    np.savez(checkpoint_path, embeddings=embeddings, next_index=next_index)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings via Gemini API.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input .jsonl file (default: corpus.jsonl)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .npy file")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .npz file (default: <output>.checkpoint.npz)")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output
    checkpoint_path = args.checkpoint or output_path.replace(".npy", "_checkpoint.npz")

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Set it in .env file.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Loading texts from {input_path}...")
    texts = load_texts(input_path)
    total = len(texts)
    print(f"Total texts: {total}")

    # Load or initialize embeddings (pre-allocated numpy array to avoid OOM)
    existing_embeddings, start_index = load_checkpoint(checkpoint_path)
    embeddings = np.zeros((total, DIMENSIONS), dtype=np.float32)
    if existing_embeddings is not None:
        assert existing_embeddings.shape == (start_index, DIMENSIONS), (
            f"Checkpoint shape mismatch: expected ({start_index}, {DIMENSIONS}), "
            f"got {existing_embeddings.shape}. Delete checkpoint and re-run."
        )
        embeddings[:start_index] = existing_embeddings

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
            if processed > 0:
                save_checkpoint(checkpoint_path, embeddings[:processed], processed)
            print("Checkpoint saved. Rerun this script to resume.")
            sys.exit(1)

        embeddings[i:batch_end] = result
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
            save_checkpoint(checkpoint_path, embeddings[:processed], processed)

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save final output
    assert embeddings.shape[0] == total, (
        f"Embedding count mismatch: expected {total}, got {embeddings.shape[0]}"
    )
    np.save(output_path, embeddings)

    # Remove checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed.")

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nEmbedding complete!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  File: {output_path} ({size_mb:.1f} MB)")
    print(f"  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

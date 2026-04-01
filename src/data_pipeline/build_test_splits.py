"""Step 5b: Build specialized test splits for evaluation scenarios.

Generates 5 test scenarios from existing processed data:
  1. standard        — existing test split as-is
  2. cross_model     — hold out all records from one generator (e.g., chatgpt)
  3. cross_platform  — hold out all records from one platform (e.g., qa_forum)
  4. adversarial     — placeholder for RAID adversarial eval data
  5. short_text      — test records with fewer than 100 words
"""

import json
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
SPLITS_PATH = os.path.join(PROCESSED_DIR, "splits.json")
RAID_EVAL_PATH = os.path.join(PROCESSED_DIR, "raid_eval.jsonl")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "test_splits.json")

HOLDOUT_MODEL = "chatgpt"
HOLDOUT_PLATFORM = "qa_forum"
SHORT_TEXT_MAX_WORDS = 100


def print_scenario_stats(name, indices, records):
    """Print label distribution for a test scenario."""
    if not indices:
        print(f"  {name}: 0 records (empty)")
        return
    labels = Counter(records[i]["label"] for i in indices)
    parts = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
    print(f"  {name}: {len(indices)} records ({parts})")


def main():
    # --- Load inputs ---
    print("=== Loading data ===")
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: {CORPUS_PATH} not found. Run preprocess.py first.")
        sys.exit(1)
    if not os.path.exists(SPLITS_PATH):
        print(f"ERROR: {SPLITS_PATH} not found. Run preprocess.py first.")
        sys.exit(1)

    corpus = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    print(f"  Corpus: {len(corpus)} records")
    print(f"  Splits: {len(splits)} entries")

    # --- 1. Standard test split ---
    print("\n=== Scenario 1: standard ===")
    standard = [i for i, r in enumerate(corpus) if splits.get(str(r["id"])) == "test"]
    print_scenario_stats("standard", standard, corpus)

    # --- 2. Cross-model: hold out all records from HOLDOUT_MODEL ---
    print(f"\n=== Scenario 2: cross_model_{HOLDOUT_MODEL} ===")
    cross_model_test = []
    cross_model_splits = {}
    for i, r in enumerate(corpus):
        rid = str(r["id"])
        if r.get("source_model") == HOLDOUT_MODEL:
            cross_model_test.append(i)
            # Exclude from train/val entirely
        else:
            orig_split = splits.get(rid)
            if orig_split in ("train", "val"):
                cross_model_splits[rid] = orig_split

    print_scenario_stats(f"cross_model_{HOLDOUT_MODEL}", cross_model_test, corpus)
    print(f"  Remaining train/val: {len(cross_model_splits)} records")

    cross_model_splits_path = os.path.join(
        PROCESSED_DIR, f"splits_cross_model_{HOLDOUT_MODEL}.json"
    )
    with open(cross_model_splits_path, "w", encoding="utf-8") as f:
        json.dump(cross_model_splits, f)
    print(f"  Saved: {cross_model_splits_path}")

    # --- 3. Cross-platform: hold out all records from HOLDOUT_PLATFORM ---
    print(f"\n=== Scenario 3: cross_platform_{HOLDOUT_PLATFORM} ===")
    cross_platform_test = []
    cross_platform_splits = {}
    for i, r in enumerate(corpus):
        rid = str(r["id"])
        if r.get("platform") == HOLDOUT_PLATFORM:
            cross_platform_test.append(i)
        else:
            orig_split = splits.get(rid)
            if orig_split in ("train", "val"):
                cross_platform_splits[rid] = orig_split

    print_scenario_stats(
        f"cross_platform_{HOLDOUT_PLATFORM}", cross_platform_test, corpus
    )
    print(f"  Remaining train/val: {len(cross_platform_splits)} records")

    cross_platform_splits_path = os.path.join(
        PROCESSED_DIR, f"splits_cross_platform_{HOLDOUT_PLATFORM}.json"
    )
    with open(cross_platform_splits_path, "w", encoding="utf-8") as f:
        json.dump(cross_platform_splits, f)
    print(f"  Saved: {cross_platform_splits_path}")

    # --- 4. Adversarial: RAID eval data ---
    print("\n=== Scenario 4: adversarial_raid ===")
    adversarial_raid = []
    if os.path.exists(RAID_EVAL_PATH):
        raid_records = []
        with open(RAID_EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                raid_records.append(json.loads(line))
        adversarial_raid = [r["id"] for r in raid_records]
        labels = Counter(r["label"] for r in raid_records)
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(labels.items()))
        print(f"  adversarial_raid: {len(adversarial_raid)} records ({parts})")
    else:
        print(f"  ⚠ {RAID_EVAL_PATH} not found — adversarial_raid will be empty")

    # --- 5. Short text: test records with < 100 words ---
    print("\n=== Scenario 5: short_text ===")
    short_text = [
        i for i in standard if len(corpus[i]["text"].split()) < SHORT_TEXT_MAX_WORDS
    ]
    print_scenario_stats("short_text", short_text, corpus)

    # --- Write output ---
    print("\n=== Writing test_splits.json ===")
    test_splits = {
        "standard": standard,
        f"cross_model_{HOLDOUT_MODEL}": cross_model_test,
        f"cross_platform_{HOLDOUT_PLATFORM}": cross_platform_test,
        "adversarial_raid": adversarial_raid,
        "short_text": short_text,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_splits, f)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  Saved: {OUTPUT_PATH} ({size_mb:.2f} MB)")

    # --- Summary ---
    print("\n=== Summary ===")
    for name, indices in test_splits.items():
        print(f"  {name}: {len(indices)} records")


if __name__ == "__main__":
    main()

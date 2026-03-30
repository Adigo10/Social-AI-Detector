"""Step 2: Preprocess raw datasets into unified corpus.jsonl."""

import glob
import json
import os
import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
RAID_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "raid_eval.jsonl")
SPLITS_PATH = os.path.join(PROCESSED_DIR, "splits.json")

RANDOM_SEED = 42

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text):
    """Replace URLs with [URL], collapse whitespace, strip."""
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub("[URL]", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def word_count(text):
    return len(text.split())


def process_multisocial():
    """Process MultiSocial CSV files."""
    print("\n--- Processing MultiSocial ---")
    csv_dir = os.path.join(RAW_DIR, "multisocial")
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")) +
                       glob.glob(os.path.join(csv_dir, "*.csv.gz")))

    if not csv_files:
        print("  WARNING: No MultiSocial CSV files found. Skipping.")
        return []

    records = []
    unknown_labels = Counter()
    for csv_path in csv_files:
        print(f"  Reading: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")

        # Discover column names (handle variations)
        text_col = next((c for c in df.columns if c.lower() in
                         ("text", "content", "post", "message")), None)
        label_col = next((c for c in df.columns if c.lower() in
                          ("label", "class", "is_ai", "generated")), None)
        model_col = next((c for c in df.columns if c.lower() in
                          ("model", "source_model", "generator", "source")), None)
        platform_col = next((c for c in df.columns if c.lower() in
                             ("platform", "social_media", "network", "source_platform")), None)

        if text_col is None:
            print(f"  WARNING: Could not find text column in {csv_path}. "
                  f"Available: {list(df.columns)}")
            continue

        print(f"  Mapped columns: text={text_col}, label={label_col}, "
              f"model={model_col}, platform={platform_col}")

        for _, row in df.iterrows():
            text = clean_text(str(row[text_col]))
            if word_count(text) < 5:
                continue

            # Determine label
            if label_col:
                raw_label = str(row[label_col]).lower().strip()
                if raw_label in ("ai", "generated", "machine", "1", "true", "yes"):
                    label = "ai"
                elif raw_label in ("human", "original", "real", "0", "false", "no"):
                    label = "human"
                else:
                    # Fallback heuristic — track unrecognized values
                    label = "ai" if "ai" in raw_label or "gen" in raw_label else "human"
                    unknown_labels[raw_label] += 1
            else:
                label = "human"

            source_model = str(row[model_col]).strip() if model_col and pd.notna(row[model_col]) else ("human" if label == "human" else "unknown_ai")
            platform = str(row[platform_col]).strip().lower() if platform_col and pd.notna(row[platform_col]) else "unknown"

            records.append({
                "text": text,
                "label": label,
                "source_model": source_model,
                "platform": platform,
                "dataset": "multisocial",
            })

    if unknown_labels:
        print(f"  WARNING: {sum(unknown_labels.values())} records had unrecognized labels "
              f"(mapped via heuristic fallback):")
        for lbl, cnt in unknown_labels.most_common(10):
            print(f"    '{lbl}': {cnt}")

    print(f"  MultiSocial records after cleaning: {len(records)}")
    return records


def process_hc3():
    """Process HC3 dataset (QA pairs with human and chatgpt answers)."""
    print("\n--- Processing HC3 ---")
    hc3_path = os.path.join(RAW_DIR, "hc3", "hc3_all.jsonl")

    if not os.path.exists(hc3_path):
        print("  WARNING: HC3 file not found. Skipping.")
        return []

    records = []
    with open(hc3_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = str(row.get("question", "")).strip()

            # Process human answers
            for answer in row.get("human_answers", []):
                text = clean_text(f"{question} {answer}")
                if word_count(text) < 5:
                    continue
                records.append({
                    "text": text,
                    "label": "human",
                    "source_model": "human",
                    "platform": "qa_forum",
                    "dataset": "hc3",
                })

            # Process ChatGPT answers
            for answer in row.get("chatgpt_answers", []):
                text = clean_text(f"{question} {answer}")
                if word_count(text) < 5:
                    continue
                records.append({
                    "text": text,
                    "label": "ai",
                    "source_model": "chatgpt",
                    "platform": "qa_forum",
                    "dataset": "hc3",
                })

    print(f"  HC3 records after cleaning: {len(records)}")
    return records


def process_raid():
    """Process RAID test CSV."""
    print("\n--- Processing RAID ---")
    raid_path = os.path.join(RAW_DIR, "raid_test.csv")

    if not os.path.exists(raid_path):
        print("  WARNING: RAID file not found. Skipping.")
        return []

    df = pd.read_csv(raid_path, low_memory=False)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Discover columns
    text_col = next((c for c in df.columns if c.lower() in
                     ("text", "generation", "content", "output")), None)
    model_col = next((c for c in df.columns if c.lower() in
                      ("model", "source_model", "generator")), None)
    label_col = next((c for c in df.columns if c.lower() in
                      ("label", "class", "is_ai")), None)

    if text_col is None:
        print(f"  WARNING: Could not find text column. Available: {list(df.columns)}")
        return []

    print(f"  Mapped columns: text={text_col}, model={model_col}, label={label_col}")

    records = []
    for _, row in df.iterrows():
        text = clean_text(str(row[text_col]))
        if word_count(text) < 5:
            continue

        # Determine label
        if label_col and pd.notna(row[label_col]):
            raw_label = str(row[label_col]).lower().strip()
            if raw_label in ("human", "0", "false", "no"):
                label = "human"
            else:
                label = "ai"
        else:
            # RAID test set: if model column exists, non-null model = AI
            if model_col and pd.notna(row[model_col]) and str(row[model_col]).strip():
                label = "ai"
            else:
                label = "human"

        source_model = str(row[model_col]).strip() if model_col and pd.notna(row[model_col]) else ("human" if label == "human" else "unknown_ai")
        records.append({
            "text": text,
            "label": label,
            "source_model": source_model,
            "platform": "raid",
            "dataset": "raid",
        })

    print(f"  RAID records after cleaning: {len(records)}")
    return records


def create_splits(records):
    """Create stratified 70/15/15 train/val/test split."""
    ids = [r["id"] for r in records]
    labels = [r["label"] for r in records]

    # First split: 70% train, 30% remainder
    train_ids, rem_ids, train_labels, rem_labels = train_test_split(
        ids, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )

    # Second split: 50/50 of remainder → 15% val, 15% test
    val_ids, test_ids, _, _ = train_test_split(
        rem_ids, rem_labels, test_size=0.50, random_state=RANDOM_SEED, stratify=rem_labels
    )

    splits = {}
    for id_ in train_ids:
        splits[str(id_)] = "train"
    for id_ in val_ids:
        splits[str(id_)] = "val"
    for id_ in test_ids:
        splits[str(id_)] = "test"

    return splits


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Starting preprocessing...")

    # Collect records from training/retrieval datasets (NOT RAID)
    all_records = []
    all_records.extend(process_multisocial())
    all_records.extend(process_hc3())

    # Process RAID separately (adversarial eval only)
    raid_records = process_raid()

    if not all_records:
        print("\nERROR: No records collected. Check that data/raw/ has downloaded files.")
        return

    # Assign sequential IDs and write corpus.jsonl (no RAID)
    print(f"\n--- Writing {OUTPUT_PATH} ---")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, record in enumerate(all_records):
            record["id"] = i
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(all_records)
    print(f"\nCorpus records: {total}")

    # Write RAID eval set separately
    if raid_records:
        print(f"\n--- Writing {RAID_OUTPUT_PATH} (adversarial eval only) ---")
        with open(RAID_OUTPUT_PATH, "w", encoding="utf-8") as f:
            for i, record in enumerate(raid_records):
                record["id"] = i
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  RAID eval records: {len(raid_records)}")

    # Create stratified 70/15/15 split
    print(f"\n--- Creating train/val/test splits (70/15/15) ---")
    splits = create_splits(all_records)

    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    split_counts = Counter(splits.values())
    for split_name in ("train", "val", "test"):
        count = split_counts.get(split_name, 0)
        print(f"  {split_name}: {count} ({100*count/total:.1f}%)")
    print(f"  Saved: {SPLITS_PATH}")

    # Statistics
    label_counts = Counter(r["label"] for r in all_records)
    dataset_counts = Counter(r["dataset"] for r in all_records)
    platform_counts = Counter(r["platform"] for r in all_records)

    print(f"\nBy label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

    print(f"\nBy dataset:")
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count} ({100*count/total:.1f}%)")

    print(f"\nBy platform:")
    for plat, count in sorted(platform_counts.items()):
        print(f"  {plat}: {count} ({100*count/total:.1f}%)")

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nOutput: {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

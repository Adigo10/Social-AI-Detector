"""Rebuild balanced test JSONL files from the evaluation standard split.

This restores the training test artifacts by subsetting the existing full test
files in corpus order. The balanced subset is defined by
processed/evaluation/test_splits.json["standard"].
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = PROJECT_ROOT / "processed" / "core"
TRAINING_DIR = PROJECT_ROOT / "processed" / "training"
EVAL_DIR = PROJECT_ROOT / "processed" / "evaluation"
BACKUP_SUFFIX = ".unbalanced_backup"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_all_test_positions() -> tuple[list[int], set[int]]:
    corpus_path = CORE_DIR / "corpus.jsonl"
    splits = load_json(CORE_DIR / "splits.json")
    standard_indices = load_json(EVAL_DIR / "test_splits.json")["standard"]
    standard_set = set(standard_indices)

    all_test_indices = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            if splits.get(str(record["id"])) == "test":
                all_test_indices.append(idx)

    selected_positions = [
        pos for pos, corpus_idx in enumerate(all_test_indices) if corpus_idx in standard_set
    ]
    return selected_positions, standard_set


def backup_file(path: Path) -> None:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"Backed up {path.name} -> {backup_path.name}")


def get_source_file(path: Path) -> Path:
    """Use the unbalanced backup as the canonical source when available."""
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    return backup_path if backup_path.exists() else path


def subset_jsonl(source: Path, positions: set[int], targets: list[Path]) -> None:
    kept_lines = []
    with source.open("r", encoding="utf-8") as f:
        for pos, line in enumerate(f):
            if pos in positions:
                kept_lines.append(line)

    for target in targets:
        with target.open("w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        print(f"Wrote {target.name}: {len(kept_lines)} lines")


def main() -> None:
    selected_positions, standard_set = get_all_test_positions()
    selected_positions_set = set(selected_positions)

    print(f"Balanced standard subset size: {len(selected_positions)}")
    print(f"Distinct corpus indices in subset: {len(standard_set)}")

    full_with_rag = TRAINING_DIR / "test_balanced_with_rag.jsonl"
    full_without_rag = TRAINING_DIR / "test_balanced_without_rag.jsonl"

    backup_file(full_with_rag)
    backup_file(full_without_rag)

    with_rag_source = get_source_file(full_with_rag)
    without_rag_source = get_source_file(full_without_rag)
    print(f"Using source for RAG: {with_rag_source.name}")
    print(f"Using source for no-RAG: {without_rag_source.name}")

    subset_jsonl(
        with_rag_source,
        selected_positions_set,
        [
            TRAINING_DIR / "test_with_rag.jsonl",
            TRAINING_DIR / "test_balanced_with_rag.jsonl",
        ],
    )
    subset_jsonl(
        without_rag_source,
        selected_positions_set,
        [
            TRAINING_DIR / "test_without_rag.jsonl",
            TRAINING_DIR / "test_balanced_without_rag.jsonl",
        ],
    )


if __name__ == "__main__":
    main()

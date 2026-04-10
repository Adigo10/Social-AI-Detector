"""Rebuild active validation and test artifacts with strict no-train-overlap.

Design:
- validation and standard test are equal-sized, balanced, and disjoint
- both carry explicit ids
- with-RAG and without-RAG use the exact same ids for each split
- cross_model_chatgpt is a subset of standard
- cross_platform_qa_forum is a subset of standard
- short_text is inactive and emitted as an empty subset

Constraints:
- exclude balanced-train ids
- exclude any exact (post_text, label) already present in balanced train
- deduplicate candidate pool by exact (text, label)
- use only original prompt-covered val/test pool so no live FAISS rebuild is needed
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 42


def detect_processed_root(project_root: Path) -> Path:
    for candidate in (project_root / "processed", project_root / "data" / "processed"):
        if (candidate / "core" / "corpus.jsonl").exists():
            return candidate
    raise FileNotFoundError("Could not find processed root with core/corpus.jsonl")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus_and_original_splits(corpus_path: Path, split_assignments: dict[str, str]):
    records_by_id: dict[int, dict] = {}
    ids_by_split: dict[str, list[int]] = {"val": [], "test": []}

    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rid = rec["id"]
            records_by_id[rid] = rec
            split_name = split_assignments.get(str(rid))
            if split_name in ids_by_split:
                ids_by_split[split_name].append(rid)

    return records_by_id, ids_by_split


def load_prompt_lookup(prompt_path: Path, ordered_ids: list[int], records_by_id: dict[int, dict]):
    lookup: dict[int, dict] = {}
    count = 0
    with prompt_path.open("r", encoding="utf-8") as f:
        for rid, line in zip(ordered_ids, f):
            row = json.loads(line)
            row["id"] = rid
            if row["output"] != records_by_id[rid]["label"]:
                raise ValueError(
                    f"Label mismatch in {prompt_path.name} for id={rid}: "
                    f"{row['output']} != {records_by_id[rid]['label']}"
                )
            lookup[rid] = row
            count += 1
    if count != len(ordered_ids):
        raise ValueError(
            f"{prompt_path.name} row count {count} does not match expected ids {len(ordered_ids)}"
        )
    return lookup


def prompt_post_text(instruction: str) -> str:
    if "Post: " in instruction:
        return instruction.rsplit("Post: ", 1)[1]
    return instruction


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_root = detect_processed_root(project_root)
    core_dir = processed_root / "core"
    training_dir = processed_root / "training"
    balanced_dir = training_dir / "balanced"
    unbalanced_dir = training_dir / "unbalanced"
    evaluation_dir = processed_root / "evaluation"
    old_training_dir = processed_root / "old_data" / "training"

    corpus_path = core_dir / "corpus.jsonl"
    splits_path = core_dir / "splits.json"
    balanced_train_path = core_dir / "balanced_train_indices.npy"

    split_assignments = load_json(splits_path)
    records_by_id, ids_by_split = load_corpus_and_original_splits(corpus_path, split_assignments)
    val_ids_old = ids_by_split["val"]
    test_ids_old = ids_by_split["test"]

    with_rag_lookup = {}
    without_rag_lookup = {}
    for split_name, ordered_ids, with_path, without_path in (
        (
            "val",
            val_ids_old,
            unbalanced_dir / "val_with_rag.jsonl",
            unbalanced_dir / "val_without_rag.jsonl",
        ),
        (
            "test",
            test_ids_old,
            old_training_dir / "test_balanced_with_rag.jsonl.unbalanced_backup",
            old_training_dir / "test_balanced_without_rag.jsonl.unbalanced_backup",
        ),
    ):
        if not with_path.exists() or not without_path.exists():
            raise FileNotFoundError(f"Missing source prompt files for {split_name}")
        with_rag_lookup.update(load_prompt_lookup(with_path, ordered_ids, records_by_id))
        without_rag_lookup.update(load_prompt_lookup(without_path, ordered_ids, records_by_id))

    balanced_train_ids = set(np.load(balanced_train_path).tolist())
    train_prompt_keys = {
        (prompt_post_text(row["instruction"]), row["output"])
        for row in read_jsonl(balanced_dir / "train_balanced_with_rag.jsonl")
    }

    candidate_records = []
    seen_text_label = set()
    for rid in val_ids_old + test_ids_old:
        if rid in balanced_train_ids:
            continue
        rec = records_by_id[rid]
        key = (rec["text"], rec["label"])
        if key in train_prompt_keys:
            continue
        if key in seen_text_label:
            continue
        seen_text_label.add(key)
        candidate_records.append(rec)

    rng = np.random.default_rng(SEED)
    humans = [rec for rec in candidate_records if rec["label"] == "human"]
    ai_chatgpt_qaforum = [
        rec
        for rec in candidate_records
        if rec["label"] == "ai"
        and rec.get("source_model") == "chatgpt"
        and rec.get("platform") == "qa_forum"
    ]
    ai_other = [
        rec
        for rec in candidate_records
        if rec["label"] == "ai"
        and not (
            rec.get("source_model") == "chatgpt"
            and rec.get("platform") == "qa_forum"
        )
    ]
    qa_forum_humans = [rec for rec in humans if rec.get("platform") == "qa_forum"]

    rng.shuffle(ai_chatgpt_qaforum)
    rng.shuffle(ai_other)
    rng.shuffle(qa_forum_humans)
    non_qaforum_humans = [rec for rec in humans if rec.get("platform") != "qa_forum"]
    rng.shuffle(non_qaforum_humans)

    split_human_count = len(humans) // 2
    special_subset_count = min(
        len(ai_chatgpt_qaforum),
        len(qa_forum_humans),
        split_human_count,
    )
    standard_qaforum_humans = qa_forum_humans[:special_subset_count]
    remaining_standard_humans = split_human_count - special_subset_count
    standard_non_qaforum_humans = non_qaforum_humans[:remaining_standard_humans]
    if len(standard_non_qaforum_humans) != remaining_standard_humans:
        raise ValueError("Insufficient non-qa_forum human records for balanced standard")

    standard_human_ids = {
        rec["id"] for rec in standard_qaforum_humans + standard_non_qaforum_humans
    }
    standard_humans = [rec for rec in humans if rec["id"] in standard_human_ids]
    val_humans = [rec for rec in humans if rec["id"] not in standard_human_ids]
    if len(standard_humans) != split_human_count or len(val_humans) != split_human_count:
        raise ValueError("Validation/test human pool did not split evenly")

    val_ai = ai_other[: split_human_count]
    ai_other = ai_other[split_human_count:]
    if len(val_ai) != split_human_count:
        raise ValueError("Insufficient AI records for balanced validation")

    selected_special_ai = ai_chatgpt_qaforum[:special_subset_count]
    selected_other_ai = ai_other[: len(standard_humans) - len(selected_special_ai)]
    if len(selected_other_ai) != len(standard_humans) - len(selected_special_ai):
        raise ValueError("Insufficient AI records for balanced standard")

    validation_records = val_humans + val_ai
    standard_records = standard_humans + selected_special_ai + selected_other_ai
    rng.shuffle(validation_records)
    rng.shuffle(standard_records)

    validation_ids = [rec["id"] for rec in validation_records]
    standard_ids = [rec["id"] for rec in standard_records]

    selected_special_ai_ids = {rec["id"] for rec in selected_special_ai}
    qa_forum_human_sample = standard_qaforum_humans
    qa_forum_human_ids = {rec["id"] for rec in qa_forum_human_sample}

    cross_platform_ids = [
        rec["id"]
        for rec in standard_records
        if rec["id"] in selected_special_ai_ids or rec["id"] in qa_forum_human_ids
    ]

    standard_human_records = [rec for rec in standard_records if rec["label"] == "human"]
    rng.shuffle(standard_human_records)
    cross_model_human_ids = [rec["id"] for rec in standard_human_records[: len(selected_special_ai)]]
    cross_model_ids = [rec["id"] for rec in selected_special_ai] + cross_model_human_ids
    rng.shuffle(cross_model_ids)

    def build_rows(ids: list[int], lookup: dict[int, dict]):
        return [
            {
                "id": rid,
                "instruction": lookup[rid]["instruction"],
                "output": lookup[rid]["output"],
            }
            for rid in ids
        ]

    val_with_rows = build_rows(validation_ids, with_rag_lookup)
    val_without_rows = build_rows(validation_ids, without_rag_lookup)
    test_with_rows = build_rows(standard_ids, with_rag_lookup)
    test_without_rows = build_rows(standard_ids, without_rag_lookup)

    val_with_rag_path = balanced_dir / "val_balanced_with_rag.jsonl"
    val_without_rag_path = balanced_dir / "val_balanced_without_rag.jsonl"
    test_with_rag_path = balanced_dir / "test_balanced_with_rag.jsonl"
    test_without_rag_path = balanced_dir / "test_balanced_without_rag.jsonl"
    write_jsonl(val_with_rag_path, val_with_rows)
    write_jsonl(val_without_rag_path, val_without_rows)
    write_jsonl(test_with_rag_path, test_with_rows)
    write_jsonl(test_without_rag_path, test_without_rows)

    active_splits = {
        "standard": standard_ids,
        "cross_model_chatgpt": cross_model_ids,
        "cross_platform_qa_forum": cross_platform_ids,
    }
    with (evaluation_dir / "test_splits.json").open("w", encoding="utf-8") as f:
        json.dump(active_splits, f, ensure_ascii=False, indent=2)

    metadata = {
        "_meta": {
            "seed": SEED,
            "candidate_pool_rule": (
                "original val/test prompt pool excluding balanced_train_indices.npy, "
                "excluding any exact (post_text,label) present in balanced train, "
                "then deduplicated by exact (text,label)"
            ),
            "candidate_counts": {
                "unique_total": len(candidate_records),
                "ai_total": sum(1 for rec in candidate_records if rec["label"] == "ai"),
                "human_total": sum(1 for rec in candidate_records if rec["label"] == "human"),
                "ai_chatgpt_qaforum": len(ai_chatgpt_qaforum),
                "human_qaforum": len(qa_forum_humans),
            },
            "scenario_counts": {
                "validation_total": len(validation_ids),
                "validation_ai": sum(1 for rec in validation_records if rec["label"] == "ai"),
                "validation_human": sum(1 for rec in validation_records if rec["label"] == "human"),
                "standard_total": len(standard_ids),
                "standard_ai": sum(1 for rec in standard_records if rec["label"] == "ai"),
                "standard_human": sum(1 for rec in standard_records if rec["label"] == "human"),
                "cross_model_total": len(cross_model_ids),
                "cross_platform_total": len(cross_platform_ids),
            },
            "artifacts": {
                "val_with_rag": str(val_with_rag_path.relative_to(project_root)),
                "val_without_rag": str(val_without_rag_path.relative_to(project_root)),
                "test_with_rag": str(test_with_rag_path.relative_to(project_root)),
                "test_without_rag": str(test_without_rag_path.relative_to(project_root)),
            },
        },
        "validation": validation_ids,
        "standard": standard_ids,
        "test": standard_ids,
        "cross_model_chatgpt": cross_model_ids,
        "cross_platform_qa_forum": cross_platform_ids,
        "short_text": [],
    }
    metadata_path = evaluation_dir / "balanced_eval_splits.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    report_lines = [
        "balanced_eval_splits",
        f"processed_root: {processed_root.relative_to(project_root)}",
        f"seed: {SEED}",
        f"candidate_pool_unique_total: {len(candidate_records)}",
        f"candidate_pool_ai_total: {sum(1 for rec in candidate_records if rec['label'] == 'ai')}",
        f"candidate_pool_human_total: {sum(1 for rec in candidate_records if rec['label'] == 'human')}",
        f"validation_total: {len(validation_ids)}",
        f"validation_ai: {sum(1 for rec in validation_records if rec['label'] == 'ai')}",
        f"validation_human: {sum(1 for rec in validation_records if rec['label'] == 'human')}",
        f"standard_total: {len(standard_ids)}",
        f"standard_ai: {sum(1 for rec in standard_records if rec['label'] == 'ai')}",
        f"standard_human: {sum(1 for rec in standard_records if rec['label'] == 'human')}",
        f"cross_model_total: {len(cross_model_ids)}",
        f"cross_platform_total: {len(cross_platform_ids)}",
        f"cross_model_subset_standard: {set(cross_model_ids).issubset(set(standard_ids))}",
        f"cross_platform_subset_standard: {set(cross_platform_ids).issubset(set(standard_ids))}",
        f"val_with_rag: {val_with_rag_path.relative_to(project_root)}",
        f"val_without_rag: {val_without_rag_path.relative_to(project_root)}",
        f"test_with_rag: {test_with_rag_path.relative_to(project_root)}",
        f"test_without_rag: {test_without_rag_path.relative_to(project_root)}",
        f"metadata_json: {metadata_path.relative_to(project_root)}",
    ]
    (evaluation_dir / "balanced_eval_splits_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("Rebuilt balanced validation and standard test artifacts.")
    for line in report_lines[1:]:
        print(line)


if __name__ == "__main__":
    main()

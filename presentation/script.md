# Presentation Script — Data Preparation
**NTU AI6130 Group G30**

---

## Slide 1 — Section Title
*~15 seconds*

> "I'll cover Data Preparation — the foundation everything else builds on. My job was to take raw datasets and produce clean, leakage-free training and evaluation artifacts for the team."

---

## Slide 2 — Datasets
*~45 seconds*

> "We work with three datasets. **MultiSocial** is the main social-text corpus — about 410,000 processed records with source tags spanning five platforms and 22 languages. It's our main source of training diversity. **HC3** adds 85,000 processed records of human and ChatGPT answers in English."
>
> "Together that's 495,000 records in the training corpus. **RAID** is kept completely separate — it's an adversarial benchmark of 671,000 eval texts, never mixed into training or validation."

---

## Slide 3 — Key Design Decisions: Datasets
*~30 seconds*

> "One thing worth calling out here: MultiSocial and HC3 use different column names across their raw files. Rather than hardcoding field mappings, I wrote auto-discovery logic so the same script ingests both datasets and produces one normalized schema. That keeps all downstream training code dataset-agnostic."

---

## Slide 4 — Pipeline Architecture
*~45 seconds*

> "The pipeline runs in five steps: download, preprocess into that unified schema, embed with Gemini Embedding 2, build a FAISS index, then assemble training files. Each step writes a persistent artifact, so the whole pipeline is resumable."
>
> "The embedding model uses the CLASSIFICATION task type. Each of the 495K texts becomes a 768-dimensional vector — 1.4 gigabytes total. The pipeline checkpoints every 10K texts."

---

## Slide 5 — Key Design Decisions: Pipeline
*~40 seconds*

> "The most important decision here is leakage prevention. The FAISS index is built on the training split only. When a val or test example queries for RAG neighbors, it can only retrieve training examples — held-out data is never indexed and can never appear as context for itself."
>
> "Every stage also writes a persistent artifact rather than keeping state in memory, so long embedding or indexing jobs can restart from checkpoints instead of from zero."

---

## Slide 6 — What Was Handed Off
*~40 seconds*

> "Here's what was produced for downstream use. The corpus, embeddings, FAISS index, and splits are the shared foundation. The training files come in four variants: with and without RAG context, standard and balanced — twelve files total."
>
> "On RAID: pre-computed embeddings were started but the run was stopped partway through. The checkpoint has 450,000 of 671,000 vectors saved and can be resumed if full adversarial evaluation is needed."

---

## Slide 7 — Key Design Decisions: Handoff
*~30 seconds*

> "The raw corpus is 80% AI, 20% human. To give the downstream team a choice, we produce a balanced 50/50 variant by undersampling, giving 133K records. Both imbalanced and balanced variants are provided — the model team decides which to train on."
>
> "The checkpointed RAID embeddings were handed off rather than blocking on full completion, so the team can resume adversarial evaluation without rerunning the work already done."

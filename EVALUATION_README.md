# Model Evaluation Script

## Overview

This script evaluates trained LLaMA/Qwen models on validation/test sets using RAID benchmark metrics. It calculates **TPR @ 1% FPR** (the primary metric) along with AUROC, Macro-F1, and Accuracy.

---

## Key Design Decisions

### 1. **Why `max_new_tokens=1`?**

The model is trained to generate only **one token** ("ai" or "human") as the classification label.

```python
# Training format
Instruction: "Classify this post: [post text]"
### Response:
ai              # ← Only 1 token!
```

**Why this works:**
- Both "ai" and "human" are verified to be **single tokens** in the tokenizer
- With `max_new_tokens=1`, the model generates exactly one token
- We capture the probability distribution over that single token
- This gives us a confidence score for ROC analysis

**If labels were multi-token:**
```python
# Example: "human" = 2 tokens
max_new_tokens=2  # Would need to generate both tokens
```

The script **fails fast** at startup if either label is not a single token.

---

### 2. **Separate Evaluation Runs (Not Resuming Training)**

**Approach:** Each evaluation creates a **new W&B run** tagged `"eval"`, separate from the training run.

**Why?**

| Approach | Training Run | Eval Runs | Comparison |
|----------|--------------|------------|------------|
| **Resume training** | `loss + eval metrics mixed` ❌ | Polluted | Hard to compare |
| **Separate runs** | `loss only` ✅ | `eval metrics only` ✅ | Easy to compare |

**Benefits:**
- ✅ Training runs stay clean (only loss curves)
- ✅ Eval runs are filterable by `tag:eval`
- ✅ Easy to compare all 6 conditions in W&B table
- ✅ No pollution of metrics

---

### 3. **Confidence Score Extraction**

**How it works:**

```python
# Generate 1 token with scores
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    output_scores=True,  # ← Key: return logits
)

# Get logits for the first (and only) generated token
first_token_logits = outputs.scores[0][0]  # [vocab_size]

# Convert to probabilities
probs = torch.softmax(first_token_logits, dim=-1)

# Extract P("ai" token)
ai_probability = probs[ai_token_id].item()
```

**Why use `outputs.scores` instead of `outputs.logits`?**

- `outputs.logits[0, -1, :]` → Last INPUT position under **teacher-forcing**
- `outputs.scores[0][0]` → First GENERATED token under **generation mode**

The latter is **correct** because:
- Uses the KV cache
- Uses generation-mode attention mask
- Matches what actually happens at inference time

---

## Usage

### Basic Evaluation

```bash
python evaluate_model.py \
    --model_path /path/to/final_model \
    --val_file val_without_rag.jsonl
```

### Test on Small Subset

```bash
python evaluate_model.py \
    --model_path /path/to/final_model \
    --val_file val_without_rag.jsonl \
    --max_samples 100
```

### Without W&B

```bash
python evaluate_model.py \
    --model_path /path/to/final_model \
    --val_file val_without_rag.jsonl \
    --no_wandb
```

---

## W&B Run Naming

### Convention

```
eval__<experiment_name>__<val_file>
```

### Examples

| Model Path | W&B Run Name |
|------------|--------------|
| `checkpoints/llama_lr0.0001_r32_ep1_20260404_191039/final_model` | `eval__llama_lr0.0001_r32_ep1_20260404_191039__val_without_rag` |
| `checkpoints/llama_lr0.00005_r32_ep1_20260405_120000/final_model` | `eval__llama_lr0.00005_r32_ep1_20260405_120000__val_without_rag` |
| `checkpoints/llama_lr0.00002_r32_ep1_20260405_140000/final_model` | `eval__llama_lr0.00002_r32_ep1_20260405_140000__val_without_rag` |

**Why this naming?**
- ✅ Includes learning rate (for ablation study comparison)
- ✅ Unique per run (timestamp)
- ✅ Filterable by `tag:eval`
- ✅ Sortable by TPR@1%FPR in W&B table

---

## Metrics Computed

### Primary Metric: TPR @ 1% FPR

**Definition:** True Positive Rate when False Positive Rate = 1%

**What it means:** "If we accept 1% of human posts as AI, how many AI posts do we catch?"

**Why this metric:**
- RAID benchmark standard
- Reflects real-world deployment (you control the false alarm rate)
- More meaningful than accuracy for imbalanced data

### Secondary Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness (can be misleading for imbalanced data) |
| **AUROC** | Area under ROC curve (overall discrimination ability) |
| **Macro F1** | Harmonic mean of precision and recall (averaged across classes) |
| **TPR** | True Positive Rate (recall for AI class) |
| **FPR** | False Positive Rate |

---

## Output Files

### 1. W&B Dashboard

**Run Name:** `eval__<experiment>__<val_file>`

**Metrics Logged:**
```python
{
    "eval/tpr_at_1fpr": 0.8234,    # PRIMARY
    "eval/accuracy": 0.8512,
    "eval/auroc": 0.9123,
    "eval/macro_f1": 0.8498,
    "eval/n_samples": 74326,
    "eval/n_ai": 37163,
    "eval/n_human": 37163,
    "eval/n_skipped": 0,
}
```

**Summary (for runs table):**
```python
wandb.summary["tpr_at_1fpr"] = 0.8234
wandb.summary["auroc"] = 0.9123
wandb.summary["macro_f1"] = 0.8498
wandb.summary["accuracy"] = 0.8512
wandb.summary["model_path"] = "..."
wandb.summary["val_file"] = "..."
```

### 2. Local JSON

**Path:** `{model_path}/eval_results_{val_stem}.json`

**Example:**
```json
{
  "eval/tpr_at_1fpr": 0.8234,
  "eval/accuracy": 0.8512,
  "eval/auroc": 0.9123,
  "eval/macro_f1": 0.8498,
  "eval/n_samples": 74326,
  "eval/n_ai": 37163,
  "eval/n_human": 37163,
  "eval/n_skipped": 0
}
```

---

## Ablation Study Workflow

### Step 1: Train with Different Learning Rates

```bash
# Train LR=1e-4
python train_with_checkpointing.py  # Creates: llama_lr0.0001_r32_ep1_<timestamp>/final_model

# Train LR=5e-5
python train_with_checkpointing.py  # Creates: llama_lr0.00005_r32_ep1_<timestamp>/final_model

# Train LR=2e-5
python train_with_checkpointing.py  # Creates: llama_lr0.00002_r32_ep1_<timestamp>/final_model
```

### Step 2: Evaluate Each Model

```bash
# Evaluate LR=1e-4
python evaluate_model.py \
    --model_path checkpoints/llama_lr0.0001_r32_ep1_<timestamp>/final_model \
    --val_file val_without_rag.jsonl

# Evaluate LR=5e-5
python evaluate_model.py \
    --model_path checkpoints/llama_lr0.00005_r32_ep1_<timestamp>/final_model \
    --val_file val_without_rag.jsonl

# Evaluate LR=2e-5
python evaluate_model.py \
    --model_path checkpoints/llama_lr0.00002_r32_ep1_<timestamp>/final_model \
    --val_file val_without_rag.jsonl
```

### Step 3: Compare in W&B

1. Go to W&B project
2. Filter by `tag:eval`
3. Sort by `tpr_at_1fpr` (descending)
4. **Pick the winner!**

Example:
```
LLaMA LR=1e-4:  TPR@1%FPR = 0.8234  ← Winner!
LLaMA LR=5e-5:  TPR@1%FPR = 0.8012
LLaMA LR=2e-5:  TPR@1%FPR = 0.7654
```

---

## Performance

### Estimated Runtime

| Dataset Size | Time (A40 GPU) |
|--------------|----------------|
| 100 samples | ~1 minute |
| 1,000 samples | ~5 minutes |
| 74,326 samples | ~2-3 hours |

### Bottlenecks

- **Model loading:** ~30 seconds (one-time)
- **Tokenization:** Done by HuggingFace `datasets` (fast)
- **Inference:** Single token generation per example (fast)
- **Metrics computation:** Negligible

### Optimization Tips

1. **Use `--max_samples` for testing** before full evaluation
2. **Run on GPU** (required - script checks for GPU)
3. **No batching needed** - single token generation is fast enough

---

## Troubleshooting

### Error: "Unsloth cannot find any torch accelerator"

**Cause:** Running on login node instead of GPU node

**Solution:** Submit via SLURM:
```bash
sbatch evaluate_slurm.sh
```

### Error: "'human' tokenizes to 2 tokens"

**Cause:** Tokenizer splits the label into multiple tokens. This means the training data labels do not match what this tokenizer expects.

**Solution:** Do NOT change `max_new_tokens`. The script raises a `ValueError` and exits intentionally. The correct fix is to verify your training data uses the exact label strings `"ai"` and `"human"` (lowercase, no punctuation), and that you are loading the same tokenizer that was used during training. If the tokenizer has changed, re-run training with the correct one.

### Warning: "unexpected first token 'xxx'"

**Cause:** Model generated unexpected token (rare for well-trained models)

**Solution:**
- Check if model training converged
- Verify training data format
- If <0.1% occurrences, safe to ignore

### High skip rate (>0.1%)

**Cause:** Many errors during evaluation

**Solution:**
- Check error logs
- Verify data format
- Check model loading

---

## Implementation Details

### Token Verification

```python
def get_label_tokens(tokenizer):
    """Verify 'ai' and 'human' are single tokens, return their IDs."""
    for label in ("ai", "human"):
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"'{label}' is {len(token_ids)} tokens, expected 1")
    ai_token_id = tokenizer("ai", add_special_tokens=False).input_ids[0]
    human_token_id = tokenizer("human", add_special_tokens=False).input_ids[0]
    return ai_token_id, human_token_id
```

**Why?** Fails fast before wasting hours on 74K examples.

### Prediction vs Confidence

```python
# Prediction: hard decision (for accuracy/F1)
if generated_id == ai_token_id:
    prediction = "ai"
elif generated_id == human_token_id:
    prediction = "human"

# Confidence: P(ai token) (for ROC/TPR@FPR)
ai_probability = probs[ai_token_id].item()
```

**Why both?**
- **Prediction** → Accuracy, F1 (need hard decisions)
- **Confidence** → ROC, TPR@FPR (need continuous scores)

---

## References

- **RAID Benchmark:** [GitHub](https://github.com/yourusername/RAID)
- **Ablation Study:** See `ABLATION_STUDY.md`
- **Training Script:** See `train_with_checkpointing.py`

---

## Summary

✅ **Correct:** Uses `max_new_tokens=1` (both labels are single tokens)
✅ **Efficient:** Single forward pass per example
✅ **Robust:** Error handling, token verification, fallback logic
✅ **Fair:** Separate eval runs for clean comparison
✅ **Standard:** RAID benchmark metrics (TPR@1%FPR primary)

**Ready for ablation study!** 🚀

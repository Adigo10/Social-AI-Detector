**AI6130 — Hyperparameter Recommendations**

SFT Ablation Study: Qwen3-32B vs LLaMA 3.1 — RAG vs No-RAG

*For Person 1 (Model Lead) — April 2026*

# **How to read this document**

This document covers every hyperparameter you need to set for your 2x3 ablation grid (6 conditions total). It is split into three sections:

* Section 1 — Fixed values. These must be identical across all 6 conditions. Changing any of them makes your comparison invalid.

* Section 2 — Tuned per condition. Run a small search (values provided) and pick the best by validation TPR@1%FPR. Document what you chose.

* Section 3 — Scope-fixed per model. Same within each model family, but may differ between Qwen3 and LLaMA due to architecture differences.

| The golden rule You are comparing model and RAG quality — not hyperparameter luck. So:   Fixed params ensure the playing field is level.   Tuned params ensure each condition gets its best possible performance.   The tuning budget (same number of search runs per condition) is what keeps it fair. |
| :---- |

# **Hyperparameter reference table**

| SECTION 1 — Fixed across all 6 conditions and both models (do not change these) |  |  |  |
| :---- | :---- | :---- | :---- |
| **Parameter** | **Recommended Value** | **Why This Value** | **Risk If Wrong** |
| **Training dataset** | **134K balanced (50/50)** | Same 66,963 AI \+ 66,963 human examples for all conditions. Any difference attributes performance to data, not RAG or model. | *Comparison invalid — you are no longer testing the same thing.* |
| **Val / test splits** | **Same frozen files** | All 6 conditions scored on identical examples. Non-negotiable for a valid ablation. | *Cherry-picking splits inflates one condition unfairly.* |
| **FAISS index** | **Balanced index, all RAG conds** | RAG conditions must retrieve from the exact same corpus. Different indexes change the evidence, not the architecture. | *RAG vs no-RAG difference becomes index quality, not RAG benefit.* |
| **LoRA rank** | **32** | Controls how many parameters LoRA adds. Same rank \= same adapter capacity for both models. Changing it changes the model, not just the LR. | *One model gets more capacity — unfair comparison.* |
| **LoRA alpha** | **64 (ratio \= 2x rank)** | Alpha controls the effective scaling of the LoRA update. Ratio of 2 is the standard recommended default across the literature. | *Wrong ratio destabilises training or makes updates too weak.* |
| **LoRA target modules** | **q\_proj, k\_proj, v\_proj, o\_proj** | Both Qwen3 and LLaMA have these 4 attention projections. Targeting only attention is sufficient for classification tasks and keeps param count low. | *Adding MLP layers increases params and VRAM significantly with little gain for detection tasks.* |
| **Sequence length** | **2048** | RAG prompts with 5 retrieved examples are long (\~1200-1500 tokens). Cutting below 2048 truncates the evidence context mid-prompt. | *Truncated RAG context \= the model cannot see all 5 examples \= unfair to RAG conditions.* |
| **Num train epochs** | **1** | Compute-constrained by NTU 12hr wall time. More importantly, both models see the same data passes — epoch count is a controlled variable. | *2 epochs risks overfitting on 134K examples at this model scale.* |
| **Effective batch size** | **32 (per\_device=4, accum=8)** | Controls gradient noise. Effective batch of 32 is large enough for stable training at this dataset size. Must match within each model family. | *Smaller batches \= noisier gradients \= more variance in results, harder to compare.* |
| **Eval & save steps** | **500 steps** | Checkpoint cadence is identical across conditions so best-checkpoint selection compares equivalent training progress. | *Coarser cadence misses the true best checkpoint.* |
| **Optimizer** | **AdamW 8-bit paged** | Standard for QLoRA fine-tuning. 8-bit reduces VRAM by \~3GB vs 32-bit AdamW with no quality loss. Paged prevents OOM spikes. | *Different optimizers change the update rule, not just the LR.* |
| **LR scheduler** | **Cosine decay** | Cosine decay gradually reduces LR toward end of training — well-suited to 1-epoch runs where you want the model to settle smoothly. | *Linear decay is harsher; constant LR leads to noisy final checkpoints.* |
| **Random seed** | **42** | Reproducibility. Both models start from the same data-shuffle and dropout state. Document this in your paper. | *Different seeds introduce variance that looks like a real performance gap.* |
| **Primary metric** | **TPR @ 1% FPR** | Your paper's evaluation protocol. Must be identical across all conditions. Changing this changes what 'better' means. | *Results not comparable to RAID leaderboard or other papers.* |
| **SECTION 2 — Tuned independently per condition (run search, pick best by val TPR@1%FPR)** |  |  |  |
| **Parameter** | **Recommended Value** | **Why This Value** | **Risk If Wrong** |
| **Learning rate (Qwen3, no RAG)** | **Try: 1e-4, 5e-5, 2e-5** | Qwen3 has deep architecture and strong pretraining. 1e-4 is often too aggressive for 32B models — 5e-5 or 2e-5 tends to be safer. No-RAG prompts are shorter so the model can afford a slightly higher LR. | *Too high: training loss spikes and never recovers. Too low: model barely moves from base weights in 1 epoch.* |
| **Learning rate (Qwen3, RAG)** | **Try: 5e-5, 2e-5, 1e-5** | RAG prompts are \~2x longer than no-RAG prompts. Longer sequences mean larger gradient norms per step, so the effective LR is already higher — compensate by starting lower. | *Too high on RAG conditions: gradient explosion from long sequences.* |
| **Learning rate (LLaMA, no RAG)** | **Try: 1e-4, 5e-5, 2e-5** | LLaMA 3.1 typically tolerates slightly higher LRs than Qwen3 at equivalent scale. Start at 1e-4 and step down. Pick by val TPR@1%FPR. | *Same risk as Qwen3 but LLaMA is generally more stable at 1e-4.* |
| **Learning rate (LLaMA, RAG)** | **Try: 5e-5, 2e-5, 1e-5** | Same reasoning as Qwen3 RAG — longer prompts need lower LR. LLaMA may tolerate 5e-5 where Qwen3 needs 2e-5. | *LLaMA \+ RAG \+ high LR: gradient norms become very large with long sequences.* |
| **Warmup ratio** | **Try: 0.03, 0.05** | Warmup gradually ramps LR from 0 to target over the first N steps. Lower LR conditions need less warmup (0.03). Higher LR conditions benefit from longer warmup (0.05) to avoid early instability. | *No warmup at LR 1e-4 often causes loss spike in first 50-100 steps that the model never fully recovers from in 1 epoch.* |
| **SECTION 3 — Scope-fixed per model family (same within Qwen conditions, may differ from LLaMA)** |  |  |  |
| **Parameter** | **Recommended Value** | **Why This Value** | **Risk If Wrong** |
| **per\_device\_train\_batch\_size** | **4 (adjust if OOM)** | Qwen3-32B and LLaMA may have slightly different VRAM profiles at 4-bit. Adjust per\_device to stay within VRAM. Key: keep effective batch \= 32 by adjusting gradient\_accumulation\_steps to compensate. | *If effective batch drops below 32, gradient estimates are noisier and the condition is not fairly compared.* |
| **Precision (bf16/fp16)** | **Auto by GPU (bf16 on A40/L40S)** | T4 only supports fp16. A40 and L40S support bf16, which is more numerically stable for large models. Unsloth handles this automatically. | *fp16 on very long sequences can produce NaN gradients — bf16 is safer.* |
| **Qwen3 thinking mode** | **enable\_thinking=False always** | Qwen3 defaults to producing \<think\> blocks before answering. This must be disabled at both training and inference time. A model trained with thinking ON and evaluated with thinking OFF will perform poorly. | *If thinking is ON during training but OFF at inference (or vice versa), the model sees a different prompt format — performance degrades significantly.* |
| **LoRA dropout** | **0.05** | Small dropout on LoRA adapters prevents overfitting. 0.05 is the standard value — low enough not to interfere with learning, high enough to regularise. | *0.0 (no dropout) can lead to adapter overfitting, especially on the in-distribution test set.* |

# **Search strategy — how many runs to do**

For each of the 4 fine-tuned conditions, run all combinations of learning rate and warmup ratio candidates below. That gives 6 runs per fine-tuned condition, 24 training runs total. Zero-shot conditions need no training runs.

| Condition | LR candidates | Warmup candidates | Total runs |
| :---- | :---- | :---- | :---- |
| Qwen3 FT, no RAG | 1e-4, 5e-5, 2e-5 | 0.03, 0.05 | 6 runs → pick best by val TPR@1%FPR |
| Qwen3 FT, RAG | 5e-5, 2e-5, 1e-5 | 0.03, 0.05 | 6 runs → pick best by val TPR@1%FPR |
| LLaMA FT, no RAG | 1e-4, 5e-5, 2e-5 | 0.03, 0.05 | 6 runs → pick best by val TPR@1%FPR |
| LLaMA FT, RAG | 5e-5, 2e-5, 1e-5 | 0.03, 0.05 | 6 runs → pick best by val TPR@1%FPR |
| *Qwen3 zero-shot* | *N/A* | *N/A* | *No training — evaluate directly* |
| *LLaMA zero-shot* | *N/A* | *N/A* | *No training — evaluate directly* |
| **TOTAL training runs** | **24** |  | Document chosen LR \+ warmup \+ val score for each |

| Why 3 LR values and not just 2? The original ablation doc uses only {1e-4, 5e-5}. This is too thin for a 32B model. 32B models are more sensitive to LR than smaller models. The difference between 5e-5 and 2e-5 can be the difference between a well-trained adapter and one that barely moved. Adding 2e-5 costs only 2 extra training runs but significantly de-risks the comparison. |
| :---- |

# **Why RAG conditions need a lower learning rate**

This is the most commonly misunderstood aspect of this setup. Here is the plain-English explanation:

**No-RAG prompt:** \~300-500 tokens. Short. The model sees: system message \+ the post to classify.

**RAG prompt:** \~1200-1600 tokens. Long. The model sees: system message \+ 5 retrieved examples with labels \+ the post to classify.

When you compute gradients during backpropagation, longer sequences produce larger gradient norms. This means that at the same learning rate, the RAG condition takes larger effective parameter update steps than the no-RAG condition. If you use the same LR for both, you are not actually applying the same training pressure — you are applying more pressure to the RAG condition.

| Practical consequence Qwen3 FT no-RAG at LR 1e-4 might train stably. Qwen3 FT RAG at LR 1e-4 might produce NaN gradients or diverge in the first 200 steps. This is not because RAG makes training harder — it is because the effective LR is too high for the longer input sequences. Dropping to 5e-5 or 2e-5 resolves it. |
| :---- |

# **What to record and report**

For each fine-tuned condition, record the following. Include this table in your paper's appendix — reviewers will ask for it.

| Condition | Best LR | Best warmup ratio | Val TPR@1%FPR | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3 FT, no RAG | \_\_\_ | \_\_\_ | \_\_\_ |  |
| Qwen3 FT, RAG | \_\_\_ | \_\_\_ | \_\_\_ |  |
| LLaMA FT, no RAG | \_\_\_ | \_\_\_ | \_\_\_ |  |
| LLaMA FT, RAG | \_\_\_ | \_\_\_ | \_\_\_ |  |

A one-line footnote in your hyperparameter table is sufficient justification: 'Each fine-tuned condition was independently tuned on the validation set; the learning rate and warmup ratio with highest val TPR@1%FPR were selected.'

# **Pre-training checklist**

* Balanced dataset confirmed: 66,963 AI \+ 66,963 human (134K total)

* Same FAISS index used for all RAG conditions

* Random seed set to 42 in both trainer and data loader

* enable\_thinking=False set in Qwen3 tokenizer at both train and eval time

* Effective batch size \= 32 verified (per\_device x gradient\_accumulation\_steps)

* eval\_steps \= save\_steps \= 500 in TrainingArguments

* save\_best\_model=True on val TPR@1%FPR (not on loss)

* Test sets not touched until final evaluation — hyperparameter selection on val set only

* Recording table above filled in before writing up results


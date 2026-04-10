## **`run_full_eval.py` — Complete Briefing**

---

## **1\. How Many Datapoints?**

**Scenarios 1-4** — sizes are fixed by whatever is in your `test_splits.json`. You do not choose these numbers, they are determined by your data pipeline. Based on your 133K dataset with 15% test split:

* Standard: \~20K (use all of it, no reason to subsample)  
* Cross-model: however many ChatGPT examples exist in your test split, maybe 2-4K  
* Cross-platform: however many QA forum examples exist, unknown until you check  
* Short text: filtered subset of standard, maybe 3-10K

**Scenario 5 (RAID)** — fixed by your `raid_eval.jsonl` file size, completely external.

**For time-constrained runs** use `--max_samples N` which caps every scenario at N. A reasonable smoke test is 200, a quick real run is 1000\. But for your final paper numbers use all available data — do not artificially cap.

---

## **2\. Should You Do All 5 Scenarios?**

Given your time constraints, here is the honest priority order:

**Must do:** Standard (Scenario 1\) and Adversarial/RAID (Scenario 5). These are your two core paper results — in-distribution performance and generalisation.

**Do if data exists:** Cross-model (Scenario 2\) and Cross-platform (Scenario 3). These are your most interesting research findings — they directly answer your research question about generalisation to unseen generators. But they only work if your data pipeline correctly built the holdout splits.

**Skip if pressed for time:** Short text (Scenario 4). It is a filtered subset of standard, least novel finding.

The script already supports skipping RAID specifically:

python run\_full\_eval.py \\  
    \--model\_path checkpoints/qwen\_rag/final \\  
    \--skip\_adversarial   \# ← skips scenario 5, saves time \+ no Gemini API needed

If a scenario is missing from `test_splits.json` the script skips it silently — so no harm done if cross-model or cross-platform data does not exist yet.

---

## **3\. Code Consistency With Proposal — Gaps**

**Gap 1 — Qwen3 thinking tokens not disabled (CRITICAL)**

Same issue as `evaluate_model.py`. The script loads the model but never sets `enable_thinking = False`. For Qwen3 this will break `max_new_tokens=1` entirely. Add this immediately after model loading:

\# Current code  
model, tokenizer \= load\_model(args.model\_path, args.max\_seq\_length)

\# Add this right after  
tokenizer.enable\_thinking \= False

And `predict_single()` uses a plain prompt format instead of chat template, same mismatch as `evaluate_model.py`:

\# Current — wrong for Qwen3  
def predict\_single(...):  
    prompt \= f"{instruction}\\n\\n\#\#\# Response:\\n"

\# Should be — matches training format  
def predict\_single(...):  
    messages \= \[  
        {"role": "system", "content": "You are a binary classifier. Respond with exactly one word: either 'ai' or 'human'."},  
        {"role": "user", "content": instruction},  
    \]  
    prompt \= tokenizer.apply\_chat\_template(  
        messages,  
        tokenize=False,  
        add\_generation\_prompt=True,  
        enable\_thinking=False,  
    )

**Gap 2 — RAID evaluated without RAG context**

Look at `build_raid_records()`:

def build\_raid\_records(raid\_path):  
    records.append({  
        "instruction": (  
            "Classify the following text as 'ai' or 'human'.\\n\\n"  
            f"Text: {rec\['text'\]}"   \# ← plain text, no retrieved examples  
        ),  
    })

Your model was trained WITH RAG context in the prompt. Evaluating it without RAG at inference time means you are testing a different input format than it was trained on. This will understate your full pipeline's performance. Either accept this limitation and note it in your report, or build a RAG-augmented version of RAID records by querying your FAISS index for each RAID text at eval time.

**The Only Remaining Question About RAID**

RAID records in `build_raid_records()` are built as plain prompts with no RAG context regardless of condition. So your RAG model gets evaluated on RAID without its expected RAG context.

But this is actually **fine and intentional** — RAID is an external benchmark with no corresponding FAISS index entries. You cannot retrieve neighbours for texts that were never embedded into your corpus. Just note this clearly in your report:

*"RAID adversarial evaluation is conducted without retrieval context as RAID texts have no corresponding entries in the training corpus index."*

That is a honest and completely defensible limitation.

**Gap 3 — Inconsistent unexpected token handling**

`run_full_eval.py` skips unexpected tokens, `evaluate_model.py` uses a probability fallback. As discussed earlier, fix `evaluate_model.py` to also skip:

\# In get\_model\_prediction\_with\_confidence()  
else:  
    return None, None  \# skip instead of fallback

\# In evaluation loop  
if prediction is None:  
    skipped \+= 1  
    continue

**Gap 4 — No ensemble**

The proposal describes combining LLM confidence with KNN majority vote. This script only evaluates the LLM alone. KNN is separate in `knn_baseline.py`. The fusion step does not exist anywhere. For your paper this means you can report LLM-only and KNN-only numbers but not the combined ensemble unless someone builds it.

**Gap 5 — Training run ID not saved**

Same issue as before — training script does not save `wandb.run.id` to a file. For `run_full_eval.py` you can pass it manually:

python run\_full\_eval.py \\  
    \--model\_path checkpoints/qwen\_rag/final \\  
    \--wandb\_run\_id abc123xyz

---

## **4\. How W\&B Logging Works**

Creates one new run per execution, named:

eval\_\_\<model\_exp\>\_\_with\_rag   or   eval\_\_\<model\_exp\>\_\_no\_rag

Tagged `["eval", "full-suite"]` so filterable separately from Phase 1 runs.

All 5 scenarios log into the same run, namespaced:

\# What gets logged  
"standard/tpr\_at\_1fpr"        : 0.82  
"standard/auroc"              : 0.91  
"cross\_model/tpr\_at\_1fpr"     : 0.74  
"cross\_platform/tpr\_at\_1fpr"  : 0.71  
"short\_text/tpr\_at\_1fpr"      : 0.68  
"adversarial/tpr\_at\_1fpr"     : 0.61

\# Per source model breakdown for RAID  
"adversarial/gpt4/tpr\_at\_1fpr"   : 0.65  
"adversarial/llama/tpr\_at\_1fpr"  : 0.58

The summary dict gets one `tpr_at_1fpr` per scenario so your W\&B runs table shows one row per model with all scenario results as columns. That is literally your paper's main results table already built.

Results also saved locally to:

checkpoints/qwen\_rag/final/eval\_results\_\_qwen\_rag\_\_with\_rag.json

---

## **Watch Out For**

* Run smoke test first with `--max_samples 200` before committing to a full run  
* Check `test_splits.json` exists and has all 4 scenario keys before submitting  
* Gap 1 (thinking tokens) will silently produce wrong results — fix this before any real run  
* RAID eval calls Gemini embedding API if using KNN predictor — make sure `GEMINI_API_KEY` is set in your SLURM job environment, or use `--skip_adversarial` to avoid it entirely since your LLM eval does not need Gemini

## **Data Argument — What's Right, What's Wrong, What To Do**

**What is right**

* Test set size is genuinely independent of training set size. That part is correct and worth saying in your report.  
* Citing compute constraints is legitimate. Reviewers accept this.  
* The proportionality myth point is real — many students wrongly think test set must be 20% of training.

---

**What is actually wrong**

1. **The MoE formula is for accuracy, not TPR@1%FPR.** The ±1.8% calculation assumes a simple binomial proportion. TPR@1%FPR is an interpolated point on a ROC curve at an extreme threshold — its variance is much higher and does not follow that formula. You cannot cite ±1.8% for your primary metric. A reviewer who knows ROC statistics will immediately call this out.

2. **"15 samples anchoring the threshold is a strength" is actually an admission of weakness.** 15 human posts defining your entire primary metric IS fragile, not stable. That is exactly the argument against 3,000, not for it.

3. **3,000 is not a universal sweet spot.** The curve flattening argument depends on metric variance. For a hard scenario like cross-model where your model is already struggling, variance is high and you need more points, not fewer.

4. **You cannot randomly subsample cross-model and cross-platform.** Those scenarios only contain examples from one held-out generator or platform. If you only have 2,000 ChatGPT test examples total, you use all 2,000 — you cannot "sample 3,000" from something that does not have 3,000.

---

**What I am proposing instead**

1. **Standard scenario** — use your full \~20K test split. No reason to subsample. Eval on 20K at max\_new\_tokens=1 on an A40 takes maybe 1-2 hours. Worth it for clean numbers.

2. **Cross-model and cross-platform** — use every single example available in those holdouts, whatever that number is. Report it honestly. The number is dictated by your data, not by a statistical target.

3. **Short text** — use all available filtered examples.

4. **RAID** — use the full external benchmark. Never subsample an external standard benchmark.

5. **For your report, say this instead:**

*"Each scenario was evaluated on all available held-out examples. Cross-model and cross-platform scenarios are bounded by the size of their respective holdout conditions. For the standard test set we evaluate on the full balanced split of N examples. Compute constraints on the NTU TC2 cluster were managed by separating training and evaluation into independent SLURM jobs."*

This is honest, defensible, and does not make a statistical claim that a reviewer can poke holes in.

**RAID PART EXPLAINED**

Sure, really simply.

---

## **Normal Eval (Scenarios 1-4)**

Every text in your test set **already exists in your corpus**. It was embedded with Gemini during data preparation and stored in FAISS. So at eval time you can look up its nearest neighbours and build the RAG prompt exactly like training:

Here are similar examples:

\- "this post sounds robotic" → ai

\- "wrote this at 3am lol" → human

Now classify this: \[test text\]

The RAG model was trained on this format, so it gets the same format at eval. ✅

---

## **RAID Eval (Scenario 5\)**

RAID texts are completely external — they were never embedded, never added to your FAISS index. So when you try to retrieve neighbours for a RAID text, **there are no neighbours to retrieve.**

You cannot build the RAG prompt. So the RAID model just gets:

Classify this: \[raid text\]

Plain, no examples, no context.

---

## **So What Does This Mean?**

Your RAG model sees a **different input format** on RAID than it was trained on. This will likely hurt its RAID score compared to what it could achieve with proper context.

But you cannot fix this without embedding all RAID texts into your FAISS index first, which is extra work. So you just note it in your report as a known limitation and move on. It actually makes for an interesting finding — does the RAG model degrade more on RAID than the no-RAG model? That gap tells you how dependent it is on retrieval context.


## **evaluate\_model.py — Complete Summary**

1\. What it is for This is your Phase 1 hyperparameter tuning tool. You run it once per training checkpoint, on your val set only, to get a single TPR@1%FPR number. You run it many times across different LR/warmup combos and pick the winner.  
2\. How it loads the model Uses Unsloth's FastLanguageModel.from\_pretrained() with 4-bit quantization. Requires a GPU — it literally crashes with a clear error message if no CUDA is detected. You must submit this as a SLURM job, not run it on the login node.  
3\. The single token verification (fail-fast) Before touching any data, it checks that both "ai" and "human" are exactly one token each in your tokenizer. If either is multi-token it raises a ValueError and exits immediately. This saves you from wasting 2-3 hours running inference on 20K examples only to get garbage results at the end.  
4\. How inference actually works For each example it generates exactly one token using model.generate() with max\_new\_tokens=1 and output\_scores=True. It then extracts the full probability distribution over the vocabulary at that one position, pulls out P("ai" token) as the confidence score, and checks the argmax token ID for the hard prediction. It never does string parsing — pure token ID comparison.  
5\. What happens with unexpected tokens If the model generates something other than "ai" or "human" as its first token, it does NOT crash or skip — it falls back to using the probability threshold (if P("ai") \> 0.5 → predict "ai"). This is different from run\_full\_eval.py which skips unexpected tokens entirely. Worth knowing because it means your accuracy/F1 could be slightly optimistic if your model is generating garbage tokens frequently.  
6\. How metrics are computed Collects all predictions and confidence scores across the full val set, then computes four metrics together at the end: TPR@1%FPR (primary), AUROC, Macro-F1, Accuracy. TPR@1%FPR is computed by building the full ROC curve and interpolating at FPR=0.01 using numpy. All four get logged.  
7\. How W\&B logging works Creates a brand new W\&B run — never resumes training run. Run is named eval\_\_\<experiment\_name\>\_\_\<val\_file\_stem\> e.g. eval\_\_qwen\_rag\_lr2e-05\_\_val\_with\_rag. Tagged "eval" so you can filter. Metrics go into both wandb.log() and wandb.summary\[\] — the summary is what appears as sortable columns in the runs table, which is your comparison view across all hyperparameter runs.  
8\. What gets saved locally Writes a JSON file directly into your model checkpoint folder named eval\_results\_\<val\_stem\>.json. So after running eval your checkpoint folder contains both the model weights and its evaluation results in one place. Useful if W\&B is down or you lose internet on the cluster.  
9\. How to run it  
bash  
\# Full val set  
python evaluate\_model.py \\  
    \--model\_path checkpoints/qwen\_rag/final\_model \\  
    \--val\_file val\_with\_rag.jsonl\\   
    \--wandb\_run\_id abc123xyz \# ← grab this from W\&B URL

\# Quick smoke test first (always do this before full run)  
python evaluate\_model.py \\  
    \--model\_path checkpoints/qwen\_rag/final\_model \\  
    \--val\_file val\_with\_rag.jsonl \\  
    \--max\_samples 100  
---

## **Gaps to Flag**

Gap 1 — Qwen3 thinking tokens not disabled Your training script explicitly sets tokenizer.enable\_thinking \= False. This eval script does not do that anywhere. If you run this against your Qwen3 checkpoint, the model may generate a thinking chain before outputting the label, which completely breaks the max\_new\_tokens=1 assumption. This needs to be added before the inference loop.  
Gap 2 — No batching Inference runs one example at a time. For 20K val examples on an A40 this is slow but manageable. If you ever need to evaluate faster, batched inference would be a significant speedup — but it is not a correctness issue, just a time issue.  
Gap 3 — Inconsistent unexpected token handling vs run\_full\_eval.py As mentioned in point 5, this script uses a probability fallback for unexpected tokens while run\_full\_eval.py skips them. This means your Phase 1 and Phase 2 metrics are not computed identically, which could cause small inconsistencies when comparing val results to final benchmark results.  
Gap 4 — Only evaluates one condition per run This script takes one val file at a time — either val\_with\_rag.jsonl or val\_without\_rag.jsonl. If you want to compare RAG vs no-RAG on the same checkpoint you need to submit two separate jobs. Not a bug, just something to be aware of when planning your SLURM submissions.

**GAP 1 DEETS for QWEN**

Good catch — let me show you exactly where it is missing by comparing the two scripts side by side.

---

## **Your Training Script — Has It**

\# Load model  
model, tokenizer \= FastLanguageModel.from\_pretrained(...)

\# CRITICAL: disable thinking tokens for Qwen3  
tokenizer.enable\_thinking \= False  \# ← RIGHT HERE, line \~55

And again in the formatting function:

tokenizer.apply\_chat\_template(  
    messages,  
    tokenize=False,  
    add\_generation\_prompt=False,  
    enable\_thinking=False,  \# ← AND HERE  
)

---

## **Your evaluate\_model.py — Missing It**

\# Load model  
model, tokenizer \= FastLanguageModel.from\_pretrained(  
    model\_name=args.model\_path,  
    max\_seq\_length=args.max\_seq\_length,  
    dtype=None,  
    load\_in\_4bit=True,  
)  
model.eval()  
\# ← NOTHING HERE. enable\_thinking is never set to False.

Then inference goes straight into:

prompt \= format\_prompt(instruction)  \# just string formatting, no chat template  
inputs \= tokenizer(prompt, return\_tensors="pt", ...)  
outputs \= model.generate(\*\*inputs, max\_new\_tokens=1, ...)

Two problems here actually. First, enable\_thinking is never disabled. Second, format\_prompt() uses a plain string format instead of the chat template, so the prompt format at eval time does not match training time at all:

def format\_prompt(instruction):  
    return f"{instruction}\\n\\n\#\#\# Response:\\n"  \# ← plain format, not chat template

But training used:

tokenizer.apply\_chat\_template(messages, enable\_thinking=False)  \# ← chat template

This mismatch is actually a bigger issue than just the thinking tokens — the model was trained on chat-formatted prompts and eval is feeding it a completely different format. This needs to be fixed by using apply\_chat\_template in eval with enable\_thinking=False to match training exactly.

**\----------------------NOW TO REMOVE SOFT PROB------------------------------------------**

## **How to Fix It — Make Both Scripts Consistent**

The cleanest fix is to make evaluate\_model.py also skip unexpected tokens, matching run\_full\_eval.py exactly. Here is the change:

**Current code (problematic):**

python  
if generated\_id \== ai\_token\_id:  
    prediction \= "ai"  
elif generated\_id \== human\_token\_id:  
    prediction \= "human"  
else:  
    decoded \= tokenizer.decode(\[generated\_id\])  
    print(f"Warning: unexpected first token '{decoded}', using prob threshold.")  
    prediction \= "ai" if ai\_probability \> 0.5 else "human"  \# ← includes bad examples

return prediction, ai\_probability

**Fixed code:**

python  
if generated\_id \== ai\_token\_id:  
    prediction \= "ai"  
elif generated\_id \== human\_token\_id:  
    prediction \= "human"  
else:  
    decoded \= tokenizer.decode(\[generated\_id\])  
    print(f"Warning: unexpected first token '{decoded}' (id={generated\_id}), skipping.")  
    return None, None  \# ← signal to caller to skip this example

Then in the evaluation loop where this function is called, handle the None return:

python  
prediction, ai\_confidence \= get\_model\_prediction\_with\_confidence(  
    model, tokenizer, instruction, ai\_token\_id, human\_token\_id  
)

\# Skip if unexpected token  
if prediction is None:  
    skipped \+= 1  
    continue

This way both scripts treat unexpected tokens identically — they are excluded from metrics entirely — and your Phase 1 and Phase 2 numbers are computed on the same basis.


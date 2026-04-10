# Local Mac Inference Tutorial

This tutorial covers the full local Apple Silicon flow for this repo:

1. Download the fine-tuned LoRA adapter weights
2. Download the gated base model
3. Merge LoRA into the base model
4. Quantize the merged model into MLX format
5. Run local inference
6. Benchmark CPU Hugging Face inference against quantized MLX inference

This path is local-only. It does not touch the TC2 cluster workflow.

## 0. Requirements

- Apple Silicon Mac
- Python 3.11 or 3.12
- Hugging Face account with access to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Local copy of this repo

The repo already contains the fine-tuned adapter in [model](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model).

Required adapter files:

- [adapter_config.json](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model/adapter_config.json)
- [adapter_model.safetensors](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model/adapter_model.safetensors)
- [tokenizer.json](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model/tokenizer.json)
- [tokenizer_config.json](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model/tokenizer_config.json)

## 1. Create a local virtual environment

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the merge and inference dependencies:

```bash
uv pip install torch transformers peft accelerate safetensors sentencepiece mlx mlx-lm
```

## 2. Download the fine-tuned adapter weights

If you already have the repo with the `model/` directory, skip this step.

If you need only the adapter artifacts from another machine, copy the whole `model/` folder into the repo root.

Expected local path:

```text
./model/
```

Quick check:

```bash
ls model
```

## 3. Download the base model

The base model is gated, so export your Hugging Face token first:

```bash
export HF_TOKEN=<your_hf_token>
```

You do not download the base model separately with a custom script. The merge step below downloads it through Transformers from:

```text
meta-llama/Meta-Llama-3.1-8B-Instruct
```

The first download is large, about 16 GB.

## 4. Merge LoRA into the base model

Run:

```bash
python scripts/merge_lora.py \
  --adapter_path model \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir model_merged
```

Script:
- [merge_lora.py](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/scripts/merge_lora.py)

What it does:

- loads tokenizer from `model/`
- downloads and loads the standard Llama 3.1 8B Instruct base model on CPU
- loads the local LoRA adapter
- merges adapter weights into the base model
- saves a plain merged checkpoint into `model_merged/`

Expected log phases:

```text
[time] Loading tokenizer from adapter path...
[time] Loading standard base model on CPU. This is the slowest step.
[time] Loading local LoRA adapter...
[time] Merging adapter into base model...
[time] Saving merged model...
[time] Saving tokenizer...
[time] Done.
```

Expected output directory:

- [model_merged](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model_merged)

Notes:

- The save path uses 2 GB shards, so the write step is more stable than one giant shard.
- This step is one-time unless the adapter changes.

## 5. Quick local CPU sanity check

Before MLX quantization, verify the merged checkpoint works on CPU:

```bash
python scripts/infer_mps.py \
  --model_path model_merged \
  --device cpu \
  --text "This is a short social media post about a new phone release."
```

Script:
- [infer_mps.py](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/scripts/infer_mps.py)

Expected output shape:

```text
Loading merged model from: ...
Device: cpu
Prediction: ai
First token: 'ai' (id=...)
P(ai): ...
P(human): ...
```

Use CPU here because the merged 8B checkpoint can exceed M2 Air MPS memory.

## 6. Quantize for MLX local inference

Run:

```bash
python scripts/quantize_mlx.py \
  --input_dir model_merged \
  --output_dir model_mlx
```

Script:
- [quantize_mlx.py](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/scripts/quantize_mlx.py)

What it does:

- reads the merged Hugging Face checkpoint from `model_merged/`
- calls `mlx_lm.convert`
- writes a quantized MLX model to `model_mlx/`

Expected log phases:

```text
[time] Input dir: ...
[time] Output dir: ...
[time] Quantized: yes
[time] Starting MLX conversion...
[time] MLX conversion finished in ...
[time] Quantized MLX model saved to: ...
```

Expected output directory:

- [model_mlx](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/model_mlx)

## 7. Run MLX inference

Use the current CLI form:

```bash
mlx_lm.generate \
  --model model_mlx \
  --prompt "Classify the following social media post as either 'ai' or 'human'. Post: Example text"
```

Expected output shape:

```text
==========
ai
==========
Prompt: 55 tokens, 4.544 tokens-per-sec
Generation: 2 tokens, 36.543 tokens-per-sec
Peak memory: 4.641 GB
```

Important:

- Do not use `python -m mlx_lm.generate ...`
- That older entry path prints a runtime warning and is deprecated

Use:

- `mlx_lm.generate ...`

or:

- `python -m mlx_lm generate ...`

## 8. Benchmark CPU HF vs quantized MLX

Run:

```bash
python scripts/benchmark_local_inference.py \
  --hf_model_path model_merged \
  --mlx_model_path model_mlx \
  --prompt "Classify the following social media post as either 'ai' or 'human'. Post: Example text"
```

Script:
- [benchmark_local_inference.py](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/scripts/benchmark_local_inference.py)

What it does:

- runs CPU inference through [infer_mps.py](/Users/nottanjune/Documents/Course-Code/Sem-2/LLM/Project/Social-AI-Detector/scripts/infer_mps.py)
- runs MLX inference through `mlx_lm.generate`
- times both runs
- parses prediction and MLX throughput output

Expected output shape:

```text
Backend comparison
- CPU HF avg seconds: ...
- MLX avg seconds: ...
CPU HF output
- Prediction: ...
- P(ai): ...
- P(human): ...
MLX output
- Prediction: ...
- Prompt TPS: ...
- Generation TPS: ...
- Peak memory: ...
```

For stabler numbers, run multiple times:

```bash
python scripts/benchmark_local_inference.py \
  --hf_model_path model_merged \
  --mlx_model_path model_mlx \
  --runs 3 \
  --prompt "Classify the following social media post as either 'ai' or 'human'. Post: Example text"
```

## 9. Full command sequence

If you want the whole local flow in one place:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install torch transformers peft accelerate safetensors sentencepiece mlx mlx-lm
export HF_TOKEN=<your_hf_token>

python scripts/merge_lora.py \
  --adapter_path model \
  --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir model_merged

python scripts/infer_mps.py \
  --model_path model_merged \
  --device cpu \
  --text "This is a short social media post about a new phone release."

python scripts/quantize_mlx.py \
  --input_dir model_merged \
  --output_dir model_mlx

mlx_lm.generate \
  --model model_mlx \
  --prompt "Classify the following social media post as either 'ai' or 'human'. Post: Example text"

python scripts/benchmark_local_inference.py \
  --hf_model_path model_merged \
  --mlx_model_path model_mlx \
  --prompt "Classify the following social media post as either 'ai' or 'human'. Post: Example text"
```

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Your local venv is missing dependencies.

Fix:

```bash
source .venv/bin/activate
uv pip install torch transformers peft accelerate safetensors sentencepiece mlx mlx-lm
```

### `ImportError` from `peft` or `transformers`

Your local Python stack is mismatched.

Use a clean venv, then reinstall the exact packages above.

### MPS out of memory

Merged 8B model is too large for your MPS memory budget.

Use:

```bash
python scripts/infer_mps.py --model_path model_merged --device cpu --text "..."
```

Then use the quantized MLX path for practical local inference.

### `python -m mlx_lm.generate` warning

This is deprecation only.

Use:

```bash
mlx_lm.generate --model model_mlx --prompt "..."
```

### `Output directory already exists` during MLX quantization

The quantization script refuses to overwrite an existing output directory.

Use a new name:

```bash
python scripts/quantize_mlx.py --input_dir model_merged --output_dir model_mlx_v2
```

Or remove the old directory yourself if you want a clean rerun.

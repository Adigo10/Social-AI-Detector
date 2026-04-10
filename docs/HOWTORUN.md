# How to Run Social-AI-Detector

## What You Need

| Asset | Where to get it |
|---|---|
| `data/processed/` directory | [Google Drive](https://drive.google.com/file/d/18tx0C_Dcg1wwGtgLJ7rr3l374sR9kOYh/view) |
| `models/llama_custom/` directory | [Google Drive](https://drive.google.com/file/d/1sN7kENlhmNFmvQ_7htD33NpKSxtl5Oxe/view?usp=sharing) |
| Gemini API key | [Google AI Studio](https://aistudio.google.com/app/apikey) |

---

## Step 1: Clone and enter the repo

```bash
git clone <repo-url>
cd Social-AI-Detector
```

## Step 2: Create a virtual environment and install dependencies

Requires Python 3.11+. Use `uv` (recommended) or plain `pip`.

```bash
# with uv
python -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# or with pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 3: Set your API key

```bash
cp .env.example .env
# then edit .env and set:
# GEMINI_API_KEY=your_key_here
```

## Step 4: Download the data and model files

1. Download the `data/processed/` archive from the link above and extract it so the path `data/processed/core/corpus.index` exists.
2. Download the `models/llama_custom/` archive and extract it so `models/llama_custom/config.json` exists.

Your project root should look like:

```
Social-AI-Detector/
  data/processed/core/
    corpus.index
    corpus.jsonl
    train_indices.npy
    embeddings.npy
  models/
    llama_custom/      # LoRA adapter weights
    llama_mlx/         # (optional) MLX-quantized model for Apple Silicon
```

## Step 5 (Mac/Apple Silicon only): Build the MLX model

Skip this step if you have a CUDA GPU or only want KNN predictions.

If `models/llama_mlx/` does not exist yet, run:

```bash
python scripts/quantize_mlx.py --input_dir models/llama_custom --output_dir models/llama_mlx
```

If `models/llama_mlx/` already exists and you want to rebuild it:

```bash
python scripts/quantize_mlx.py --input_dir models/llama_custom --output_dir models/llama_mlx --force
```

This requires `mlx` and `mlx-lm` installed:

```bash
uv pip install mlx mlx-lm
```

## Step 6: Start the server

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` (KNN index load takes ~20 seconds).

## Step 7: Open the UI

Go to **http://localhost:8000** in your browser.

API docs are at **http://localhost:8000/docs**.

---

## Model availability

| Model | Requirement | Fallback behavior |
|---|---|---|
| `knn` | `GEMINI_API_KEY` + FAISS index | Unavailable if either is missing |
| `llm` (Llama 3.1 8B) | CUDA GPU (4-bit BNB) OR MLX model at `models/llama_mlx/` | Falls back down the chain; unavailable if neither works |
| `ensemble` | Both `knn` and `llm` available | Degrades to KNN-only if LLM is unavailable |

The server always starts. Unavailable models are reported in `/api/health` and show as disabled in the UI.

---

## Troubleshooting

**`GEMINI_API_KEY not set`**: Make sure `.env` exists at the project root with the key set.

**`corpus.index not found`**: Extract the `data/processed/` archive so the path `data/processed/core/corpus.index` exists.

**`LlamaDetector: peft import failed`**: Your `transformers` version is incompatible with `peft`. The server will still start in KNN-only mode. To fix: `uv pip install "transformers<5.0"`.

**`FileExistsError` from `quantize_mlx.py`**: Pass `--force` to overwrite the existing output directory.

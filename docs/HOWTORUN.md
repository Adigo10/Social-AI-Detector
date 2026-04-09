# How to Run Social-AI-Detector

## Prerequisites

- `GEMINI_API_KEY` in a `.env` file at the project root
- FAISS index built at `data/processed/core/corpus.index`

- /data/processed (Yu's Files) link: https://drive.google.com/file/d/18tx0C_Dcg1wwGtgLJ7rr3l374sR9kOYh/view

- /models/llama_custom (Tanmay & Twissa Model files) link: https://drive.google.com/file/d/1sN7kENlhmNFmvQ_7htD33NpKSxtl5Oxe/view?usp=sharing

## Setup


# 1. Activate the virtual environment
venv or conda

# 2. Install API dependencies (first time only)
pip install fastapi "uvicorn[standard]"



## Run the Server

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Wait for `Application startup complete.` in the terminal (KNN index load takes ~20s).

## Open the UI

Go to **http://localhost:8000** in your browser.

## Notes

- **KNN** requires `GEMINI_API_KEY` and the FAISS index.
- **LLM** (Llama 3.1 8B) requires a CUDA GPU with `torch`, `peft`, `bitsandbytes` installed.
- If neither GPU nor index is available, the server still starts — affected models show as unavailable.
- API docs: **http://localhost:8000/docs**

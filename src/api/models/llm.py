"""LlamaDetector: Llama 3.1 8B + LoRA adapter for AI-text classification.

Device fallback chain (tried in order):
  1. CUDA  — 4-bit BNB quantized (fastest, requires NVIDIA GPU + bitsandbytes)
  2. MLX   — Apple Silicon native via mlx_lm (requires mlx_lm + models/llama_mlx/)
  3. CPU   — standard fp32 transformers (slow, ~minutes/text, works everywhere)

If all three paths fail, is_available() returns False and the server starts in
KNN-only mode via EnsembleDetector's graceful degradation.

Prompt format matches the training data built by
src/data_pipeline/build_training_data.py — reuses build_rag_instruction() and
build_plain_instruction() directly to avoid drift.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseDetector

# So we can import from src/data_pipeline without installing as a package
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_pipeline.build_training_data import (  # noqa: E402
    build_plain_instruction,
    build_rag_instruction,
)

K_RAG_NEIGHBORS = 5   # how many neighbors to pass as RAG context


def build_rag_pairs_from_neighbors(
    neighbors: List[Dict[str, Any]],
    limit: int = K_RAG_NEIGHBORS,
) -> List[tuple[str, str]]:
    """Prefer full retrieved text so runtime prompt matches training format."""
    pairs: List[tuple[str, str]] = []
    for neighbor in neighbors[:limit]:
        text = neighbor.get("full_text") or neighbor.get("text_snippet") or ""
        pairs.append((text, neighbor["label"]))
    return pairs


class LlamaDetector(BaseDetector):
    """Fine-tuned Llama 3.1 8B (LoRA) classifier with CUDA → MLX → CPU fallback."""

    def __init__(self, adapter_path: str, mlx_path: str = ""):
        self._adapter_path = adapter_path
        self._mlx_path = mlx_path
        self._available = False
        self._device = "none"
        self._model = None
        self._tokenizer = None
        self._ai_token_id: int = -1
        self._human_token_id: int = -1

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # ------------------------------------------------------------------
            # Path 1: CUDA — 4-bit BNB quantized (unsloth variant)
            # ------------------------------------------------------------------
            if torch.cuda.is_available():
                from peft import PeftModel
                from transformers import BitsAndBytesConfig

                print("LlamaDetector: using device=cuda")
                tokenizer = AutoTokenizer.from_pretrained(adapter_path)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                base_model_id = "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"LlamaDetector: loading base model {base_model_id} ...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
                print("LlamaDetector: loading LoRA adapter ...")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model.eval()

                self._ai_token_id = tokenizer.encode("ai", add_special_tokens=False)[0]
                self._human_token_id = tokenizer.encode("human", add_special_tokens=False)[0]
                self._model = model
                self._tokenizer = tokenizer
                self._device = "cuda"
                self._available = True
                print("LlamaDetector: ready (cuda).")
                return

            # ------------------------------------------------------------------
            # Path 2: MLX — Apple Silicon native (mlx_lm + converted model)
            # ------------------------------------------------------------------
            try:
                import mlx_lm
                import mlx.core as mx  # noqa: F401 — validates mlx is installed

                mlx_model_path = Path(self._mlx_path)
                if not mlx_model_path.exists():
                    raise FileNotFoundError(
                        f"MLX model directory not found at {self._mlx_path}. "
                        "Run scripts/quantize_mlx.py to create it."
                    )

                print(f"LlamaDetector: loading MLX model from {self._mlx_path} ...")
                mlx_model, mlx_tokenizer = mlx_lm.load(self._mlx_path)

                self._ai_token_id = mlx_tokenizer.encode("ai", add_special_tokens=False)[0]
                self._human_token_id = mlx_tokenizer.encode("human", add_special_tokens=False)[0]
                self._model = mlx_model
                self._tokenizer = mlx_tokenizer
                self._device = "mlx"
                self._available = True
                print("LlamaDetector: ready (mlx).")
                return

            except (ImportError, FileNotFoundError) as mlx_err:
                print(f"LlamaDetector: MLX unavailable ({mlx_err}), falling back to CPU.")

            # ------------------------------------------------------------------
            # Path 3: CPU — fp32 transformers (slow but works everywhere)
            # ------------------------------------------------------------------
            try:
                from peft import PeftModel
            except ImportError as ie:
                print(f"LlamaDetector: peft import failed ({ie}), CPU path unavailable.")
                return

            print("LlamaDetector: WARNING — CPU inference will be very slow (~minutes/text).")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            print(f"LlamaDetector: loading base model {base_model_id} on CPU ...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            print("LlamaDetector: loading LoRA adapter ...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()

            self._ai_token_id = tokenizer.encode("ai", add_special_tokens=False)[0]
            self._human_token_id = tokenizer.encode("human", add_special_tokens=False)[0]
            self._model = model
            self._tokenizer = tokenizer
            self._device = "cpu"
            self._available = True
            print("LlamaDetector: ready (cpu — slow).")

        except Exception as e:
            print(f"LlamaDetector: failed to load — {e}")

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return f"Llama 3.1 8B (LoRA) fine-tuned on RAG instruction format — device={self._device}"

    def is_available(self) -> bool:
        return self._available

    def predict(
        self,
        texts: List[str],
        neighbors: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for idx, text in enumerate(texts):
            try:
                nbrs = (neighbors[idx] if neighbors is not None and idx < len(neighbors)
                        else None)
                if nbrs:
                    rag_pairs = build_rag_pairs_from_neighbors(nbrs)
                    instruction = build_rag_instruction(text, rag_pairs)
                else:
                    instruction = build_plain_instruction(text)

                if self._device == "mlx":
                    result = self._predict_mlx(instruction)
                else:
                    result = self._predict_torch(instruction)

                results.append(result)

            except Exception as e:
                print(f"LlamaDetector.predict error on text {idx}: {e}")
                results.append({
                    "prediction": "human",
                    "confidence": 0.5,
                    "neighbors": [],
                    "knn_confidence": None,
                    "llm_confidence": 0.5,
                    "alpha_used": 0.0,
                })

        return results

    def _predict_torch(self, instruction: str) -> Dict[str, Any]:
        """Inference via HuggingFace generate — handles both cuda and cpu paths."""
        import torch
        import torch.nn.functional as F

        messages = [{"role": "user", "content": instruction}]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=3,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
                do_sample=False,
            )

        first_logits = out.scores[0][0]   # shape: (vocab_size,)
        ai_logit = first_logits[self._ai_token_id].item()
        human_logit = first_logits[self._human_token_id].item()

        probs = F.softmax(
            torch.tensor([ai_logit, human_logit], dtype=torch.float32),
            dim=0,
        )
        ai_prob = float(probs[0])   # P(ai)

        # Determine prediction based on which token has higher logit
        is_ai = ai_logit > human_logit
        prediction = "ai" if is_ai else "human"

        # Confidence should be the probability of the predicted class
        confidence = round(ai_prob if is_ai else (1 - ai_prob), 4)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "neighbors": [],
            "knn_confidence": None,
            "llm_confidence": ai_prob,
            "alpha_used": 0.0,
        }

    def _predict_mlx(self, instruction: str) -> Dict[str, Any]:
        """Inference via mlx_lm forward pass — Apple Silicon native path."""
        import mlx.core as mx

        messages = [{"role": "user", "content": instruction}]
        tokens = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,   # plain Python list of ints
        )

        input_array = mx.array([tokens])          # shape: (1, seq_len)
        logits = self._model(input_array)          # shape: (1, seq_len, vocab_size)
        mx.eval(logits)                            # force lazy evaluation before indexing

        last_logits = logits[0, -1, :]             # next-token distribution
        ai_logit = float(last_logits[self._ai_token_id])
        human_logit = float(last_logits[self._human_token_id])

        # Numerically stable two-token softmax
        max_l = max(ai_logit, human_logit)
        exp_ai = math.exp(ai_logit - max_l)
        exp_human = math.exp(human_logit - max_l)
        ai_prob = exp_ai / (exp_ai + exp_human)   # P(ai)

        # Determine prediction based on which token has higher probability
        is_ai = ai_logit > human_logit
        prediction = "ai" if is_ai else "human"

        # Confidence should be the probability of the predicted class
        confidence = round(ai_prob if is_ai else (1 - ai_prob), 4)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "neighbors": [],
            "knn_confidence": None,
            "llm_confidence": ai_prob,
            "alpha_used": 0.0,
        }

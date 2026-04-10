"""LlamaDetector: Llama 3.1 8B + LoRA adapter for AI-text classification.

Loads the fine-tuned model from models/llama_custom/ (LoRA adapter on top of
unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit).

Requires: GPU + torch + transformers + peft + bitsandbytes.
If any requirement is missing or no GPU is found, is_available() returns False
and the server starts cleanly in KNN-only mode.

Prompt format matches the training data built by
src/data_pipeline/build_training_data.py — reuses build_rag_instruction() and
build_plain_instruction() directly to avoid drift.
"""

import sys
import os
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


class LlamaDetector(BaseDetector):
    """Fine-tuned Llama 3.1 8B (4-bit LoRA) classifier."""

    def __init__(self, adapter_path: str):
        self._adapter_path = adapter_path
        self._available = False
        self._model = None
        self._tokenizer = None
        self._ai_token_id: int = -1
        self._human_token_id: int = -1

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                print("LlamaDetector: no CUDA or MPS device detected — skipping model load.")
                return

            print(f"LlamaDetector: using device={device}")
            print(f"LlamaDetector: loading tokenizer from {adapter_path} ...")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            if device == "cuda":
                base_model_id = "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
            else:
                # MPS: bitsandbytes not supported; use standard fp16 base model
                base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                load_kwargs = {"torch_dtype": torch.float16, "device_map": {"": device}}

            print(f"LlamaDetector: loading base model {base_model_id} ...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                **load_kwargs,
            )

            print("LlamaDetector: loading LoRA adapter ...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()

            # Cache token IDs for "ai" and "human" (used for confidence extraction)
            # encode() without special tokens; take the first sub-token if multi-token
            self._ai_token_id = tokenizer.encode("ai", add_special_tokens=False)[0]
            self._human_token_id = tokenizer.encode("human", add_special_tokens=False)[0]

            self._model = model
            self._tokenizer = tokenizer
            self._available = True
            print("LlamaDetector: ready.")

        except Exception as e:
            print(f"LlamaDetector: failed to load — {e}")

    @property
    def name(self) -> str:
        return "llm"

    @property
    def description(self) -> str:
        return "Llama 3.1 8B (4-bit LoRA) fine-tuned on RAG instruction format"

    def is_available(self) -> bool:
        return self._available

    def predict(
        self,
        texts: List[str],
        neighbors: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        results = []
        for idx, text in enumerate(texts):
            try:
                # --- Build prompt ---
                nbrs = (neighbors[idx] if neighbors is not None and idx < len(neighbors)
                        else None)
                if nbrs:
                    # Convert neighbor dicts to (text, label) tuples expected by builder
                    rag_pairs = [(n["text_snippet"], n["label"]) for n in nbrs[:K_RAG_NEIGHBORS]]
                    instruction = build_rag_instruction(text, rag_pairs)
                else:
                    instruction = build_plain_instruction(text)

                messages = [{"role": "user", "content": instruction}]
                input_ids = self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self._model.device)

                # --- Generate (max 3 tokens — "ai" or "human" are short) ---
                with torch.no_grad():
                    out = self._model.generate(
                        input_ids,
                        max_new_tokens=3,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self._tokenizer.pad_token_id,
                        do_sample=False,
                    )

                # --- Confidence from first token logits ---
                first_logits = out.scores[0][0]   # shape: (vocab_size,)
                ai_score = first_logits[self._ai_token_id].item()
                human_score = first_logits[self._human_token_id].item()
                probs = F.softmax(
                    torch.tensor([ai_score, human_score], dtype=torch.float32), dim=0
                )
                confidence = float(probs[0])   # P(ai)

                # --- Parse generated text as prediction ---
                generated = self._tokenizer.decode(
                    out.sequences[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip().lower()

                # Trust logits for close calls; use text parse for clear outputs
                if confidence >= 0.5:
                    prediction = "ai"
                elif generated.startswith("human"):
                    prediction = "human"
                else:
                    prediction = "human"  # safe default

                results.append({"prediction": prediction, "confidence": confidence, "neighbors": []})

            except Exception as e:
                print(f"LlamaDetector.predict error on text {idx}: {e}")
                results.append({"prediction": "human", "confidence": 0.5, "neighbors": []})

        return results

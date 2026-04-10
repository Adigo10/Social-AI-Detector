#!/usr/bin/env python3
"""One-time LoRA merge for local MPS-friendly inference.

Loads the standard Llama 3.1 8B Instruct base model, applies the local LoRA
adapter from this repo, merges the weights, and saves a plain merged checkpoint.
The merged checkpoint can then be loaded on Apple Silicon without Unsloth,
bitsandbytes, or PEFT at inference time.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge local LoRA into a standard base model.")
    parser.add_argument(
        "--adapter_path",
        default="model",
        help="Directory containing adapter_model.safetensors and adapter_config.json.",
    )
    parser.add_argument(
        "--base_model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF repo id for the standard full-precision base model.",
    )
    parser.add_argument(
        "--output_dir",
        default="model_merged",
        help="Directory to write the merged model and tokenizer.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype used while loading and saving the merged base model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    hf_token = resolve_token(args.hf_token)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    if not (adapter_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {adapter_path}")
    if hf_token is None:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, "
            "or pass --hf_token explicitly."
        )

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing local merge dependencies. Install with:\n"
            "uv pip install torch transformers peft accelerate safetensors sentencepiece"
        ) from exc

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    log(f"Adapter path: {adapter_path}")
    log(f"Base model: {args.base_model}")
    log(f"Output dir: {output_dir}")
    log(f"Dtype: {args.torch_dtype}")
    log("Loading tokenizer from adapter path...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), token=hf_token, use_fast=True)
    log(f"Tokenizer loaded in {time.time() - start:.1f}s")

    log("Loading standard base model on CPU. This is the slowest step.")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=hf_token,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    log(f"Base model loaded in {time.time() - start:.1f}s")

    log("Loading local LoRA adapter...")
    start = time.time()
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    log(f"LoRA adapter loaded in {time.time() - start:.1f}s")

    log("Merging adapter into base model...")
    start = time.time()
    merged_model = peft_model.merge_and_unload()
    log(f"Merge completed in {time.time() - start:.1f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    log("Saving merged model...")
    start = time.time()
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    log(f"Merged weights saved in {time.time() - start:.1f}s")
    log("Saving tokenizer...")
    start = time.time()
    tokenizer.save_pretrained(str(output_dir))
    log(f"Tokenizer saved in {time.time() - start:.1f}s")

    log("Done.")
    log(f"Merged checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()

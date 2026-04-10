#!/usr/bin/env python3
"""Convert a merged HF checkpoint into a quantized MLX model for local Mac inference."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize an existing merged HF checkpoint into MLX format."
    )
    parser.add_argument(
        "--input_dir",
        default="model_merged",
        help="Path to the merged Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="models/llama_mlx",
        help="Path to write the quantized MLX model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    parser.add_argument(
        "--no_quantize",
        action="store_true",
        help="Convert to MLX without quantization.",
    )
    return parser.parse_args()


def require_mlx_lm() -> None:
    if importlib.util.find_spec("mlx_lm") is None:
        raise RuntimeError(
            "Missing MLX conversion dependencies. Install inside your local venv with:\n"
            "uv pip install mlx mlx-lm"
        )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Merged model not found at {input_dir}. Run scripts/merge_lora.py first."
        )
    if output_dir.exists():
        if args.force:
            import shutil
            log(f"--force: removing existing output directory {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --force to overwrite or pick a new --output_dir."
            )

    require_mlx_lm()

    convert_cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--hf-path",
        str(input_dir),
        "--mlx-path",
        str(output_dir),
    ]
    if not args.no_quantize:
        convert_cmd.append("-q")

    log(f"Input dir: {input_dir}")
    log(f"Output dir: {output_dir}")
    log(f"Quantized: {'no' if args.no_quantize else 'yes'}")
    log(f"Python: {sys.executable}")
    log("Starting MLX conversion...")
    start = time.time()
    subprocess.run(convert_cmd, check=True)
    elapsed = time.time() - start
    log(f"MLX conversion finished in {elapsed:.1f}s")

    if not output_dir.exists():
        raise RuntimeError(f"MLX conversion reported success but {output_dir} was not created.")

    log(f"Quantized MLX model saved to: {output_dir}")
    log("Next step:")
    log(f'python -m mlx_lm.generate --model "{output_dir}" --prompt "Classify this post as ai or human."')


if __name__ == "__main__":
    main()

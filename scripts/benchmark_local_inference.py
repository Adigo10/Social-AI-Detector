#!/usr/bin/env python3
"""Benchmark local CPU HF inference against quantized MLX inference."""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark merged CPU inference against quantized MLX inference."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt or raw classification text to run through both paths.",
    )
    parser.add_argument(
        "--hf_model_path",
        default="model_merged",
        help="Path to the merged HF checkpoint used by scripts/infer_mps.py.",
    )
    parser.add_argument(
        "--mlx_model_path",
        default="model_mlx",
        help="Path to the quantized MLX model directory.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per backend.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Max sequence length for the CPU HF path.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> tuple[float, str, str]:
    start = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - start
    return elapsed, completed.stdout, completed.stderr


def parse_cpu_output(stdout: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in stdout.splitlines():
        if line.startswith("Prediction: "):
            data["prediction"] = line.split(": ", 1)[1].strip()
        elif line.startswith("P(ai): "):
            data["p_ai"] = line.split(": ", 1)[1].strip()
        elif line.startswith("P(human): "):
            data["p_human"] = line.split(": ", 1)[1].strip()
    return data


def parse_mlx_output(stdout: str) -> dict[str, str]:
    data: dict[str, str] = {}
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    if len(lines) >= 3 and lines[-3] == "==========":
        data["prediction"] = lines[-2].strip()
    for line in lines:
        if line.startswith("Prompt: "):
            data["prompt_stats"] = line
            match = re.search(r"([0-9.]+) tokens-per-sec", line)
            if match:
                data["prompt_tps"] = match.group(1)
        elif line.startswith("Generation: "):
            data["generation_stats"] = line
            match = re.search(r"([0-9.]+) tokens-per-sec", line)
            if match:
                data["generation_tps"] = match.group(1)
        elif line.startswith("Peak memory: "):
            data["peak_memory"] = line.split(": ", 1)[1].strip()
    return data


def benchmark_cpu(prompt: str, model_path: Path, runs: int, max_seq_length: int) -> dict[str, object]:
    timings = []
    parsed: dict[str, str] = {}
    last_stdout = ""
    for _ in range(runs):
        elapsed, stdout, _stderr = run_command(
            [
                sys.executable,
                "scripts/infer_mps.py",
                "--model_path",
                str(model_path),
                "--device",
                "cpu",
                "--text",
                prompt,
                "--max_seq_length",
                str(max_seq_length),
            ]
        )
        timings.append(elapsed)
        parsed = parse_cpu_output(stdout)
        last_stdout = stdout
    return {
        "backend": "cpu_hf",
        "avg_seconds": statistics.mean(timings),
        "runs": runs,
        "parsed": parsed,
        "raw_stdout": last_stdout,
    }


def benchmark_mlx(prompt: str, model_path: Path, runs: int) -> dict[str, object]:
    timings = []
    parsed: dict[str, str] = {}
    last_stdout = ""
    for _ in range(runs):
        elapsed, stdout, _stderr = run_command(
            [
                "mlx_lm.generate",
                "--model",
                str(model_path),
                "--prompt",
                prompt,
            ]
        )
        timings.append(elapsed)
        parsed = parse_mlx_output(stdout)
        last_stdout = stdout
    return {
        "backend": "mlx_quantized",
        "avg_seconds": statistics.mean(timings),
        "runs": runs,
        "parsed": parsed,
        "raw_stdout": last_stdout,
    }


def print_summary(cpu_result: dict[str, object], mlx_result: dict[str, object]) -> None:
    print("Backend comparison")
    print(f"- CPU HF avg seconds: {cpu_result['avg_seconds']:.3f}")
    print(f"- MLX avg seconds: {mlx_result['avg_seconds']:.3f}")

    cpu_parsed = cpu_result["parsed"]
    mlx_parsed = mlx_result["parsed"]
    assert isinstance(cpu_parsed, dict)
    assert isinstance(mlx_parsed, dict)

    print("CPU HF output")
    print(f"- Prediction: {cpu_parsed.get('prediction', 'unknown')}")
    print(f"- P(ai): {cpu_parsed.get('p_ai', 'n/a')}")
    print(f"- P(human): {cpu_parsed.get('p_human', 'n/a')}")

    print("MLX output")
    print(f"- Prediction: {mlx_parsed.get('prediction', 'unknown')}")
    print(f"- Prompt TPS: {mlx_parsed.get('prompt_tps', 'n/a')}")
    print(f"- Generation TPS: {mlx_parsed.get('generation_tps', 'n/a')}")
    print(f"- Peak memory: {mlx_parsed.get('peak_memory', 'n/a')}")


def main() -> None:
    args = parse_args()
    hf_model_path = Path(args.hf_model_path).expanduser().resolve()
    mlx_model_path = Path(args.mlx_model_path).expanduser().resolve()

    if not hf_model_path.exists():
        raise FileNotFoundError(f"Missing merged HF model: {hf_model_path}")
    if not mlx_model_path.exists():
        raise FileNotFoundError(f"Missing MLX model: {mlx_model_path}")

    cpu_result = benchmark_cpu(args.prompt, hf_model_path, args.runs, args.max_seq_length)
    mlx_result = benchmark_mlx(args.prompt, mlx_model_path, args.runs)
    print_summary(cpu_result, mlx_result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bench = load_module("benchmark_local_inference", "scripts/benchmark_local_inference.py")
infer_mps = load_module("infer_mps", "scripts/infer_mps.py")


def test_parse_mlx_output_extracts_prediction_before_stats():
    stdout = """==========
'human'
==========
Prompt: 171 tokens, 12.285 tokens-per-sec
Generation: 4 tokens, 16.888 tokens-per-sec
Peak memory: 4.779 GB
"""

    parsed = bench.parse_mlx_output(stdout)

    assert parsed["prediction"] == "human"
    assert parsed["prompt_tps"] == "12.285"
    assert parsed["generation_tps"] == "16.888"
    assert parsed["peak_memory"] == "4.779 GB"


def test_build_cpu_command_uses_instruction_mode_without_double_wrapping():
    cmd = bench.build_cpu_command(
        input_value="Classify the following social media post as either 'ai' or 'human'.\n\nPost: Hello world",
        input_mode="instruction",
        model_path=Path("model_merged"),
        max_seq_length=2048,
    )

    assert "--instruction" in cmd
    assert "--text" not in cmd


def test_build_mlx_command_matches_hf_prompting_contract_for_instruction_mode():
    instruction = "Classify the following social media post as either 'ai' or 'human'.\n\nPost: Hello world"

    cmd = bench.build_mlx_command(
        input_value=instruction,
        input_mode="instruction",
        model_path=Path("model_mlx"),
    )

    prompt_index = cmd.index("--prompt")
    system_index = cmd.index("--system-prompt")

    assert cmd[prompt_index + 1] == instruction
    assert cmd[system_index + 1] == infer_mps.SYSTEM_PROMPT
    assert "--max-tokens" in cmd
    assert cmd[cmd.index("--max-tokens") + 1] == "1"
    assert "--temp" in cmd
    assert cmd[cmd.index("--temp") + 1] == "0"

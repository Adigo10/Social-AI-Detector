"""Fine-tune Llama 3.1 8B Instruct with QLoRA using Unsloth.

Uses RAG-augmented instruction data for AI-generated text detection.
Supports two modes:
  - No-checkpoint mode (--no-checkpoint): trains without saving checkpoints,
    intended for demonstrating wall-time requirements to cluster admins.
  - Checkpoint mode (default): saves checkpoints every N steps with SIGTERM
    handling for graceful shutdown under SLURM wall-time limits.

Usage:
    # Phase 1: no-checkpoint run (6-hour QoS evidence)
    python src/training/train_llama_rag.py --config configs/llama_rag.yaml --no-checkpoint

    # Phase 2: full training with checkpoints (12-hour QoS)
    python src/training/train_llama_rag.py --config configs/llama_rag.yaml

    # Resume from last checkpoint
    python src/training/train_llama_rag.py --config configs/llama_rag.yaml --resume
"""

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import time

import unsloth  # must be imported before transformers
import torch
import yaml
from datasets import load_dataset
from transformers import TrainerCallback


def log(message):
    """Emit a timestamped log line and flush immediately for SLURM logs."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def setup_verbose_logging():
    """Enable verbose library logging and fault diagnostics."""
    faulthandler.enable(all_threads=True)

    # Default to verbose diagnostics unless explicitly overridden by the job script.
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "debug")
    os.environ.setdefault("HF_DEBUG", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    os.environ.setdefault("DATASETS_VERBOSITY", "debug")
    os.environ.setdefault("WANDB_CONSOLE", "wrap")
    os.environ.setdefault("WANDB_WATCH", "false")

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Reduce the noisiest network stack logs while keeping HF-level debug output.
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.INFO)

    try:
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_debug()
    except Exception as exc:
        log(f"Unable to enable huggingface_hub debug logging: {exc}")

    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_info()
        transformers_logging.enable_default_handler()
        transformers_logging.enable_explicit_format()
    except Exception as exc:
        log(f"Unable to enable transformers verbose logging: {exc}")

    try:
        from datasets.utils import logging as datasets_logging
        datasets_logging.set_verbosity_debug()
    except Exception as exc:
        log(f"Unable to enable datasets verbose logging: {exc}")


class VerboseLoggingCallback(TrainerCallback):
    """Emit trainer lifecycle and metrics logs in plain text."""

    def __init__(self, wandb_run_id_file=None, is_resume=False):
        self.wandb_run_id_file = wandb_run_id_file
        self.is_resume = is_resume

    def on_train_begin(self, args, state, control, **kwargs):
        log("Trainer callback: training loop started")
        # Save W&B run ID early (Trainer has initialised wandb by this point)
        # so that a wall-time kill doesn't lose the run ID for future resumes.
        if self.wandb_run_id_file and not self.is_resume:
            try:
                import wandb as _wandb
                if _wandb.run is not None and not os.path.exists(self.wandb_run_id_file):
                    os.makedirs(os.path.dirname(self.wandb_run_id_file), exist_ok=True)
                    with open(self.wandb_run_id_file, "w") as _f:
                        _f.write(_wandb.run.id)
                    log(f"W&B run id saved early: {_wandb.run.id} -> {self.wandb_run_id_file}")
            except Exception as _exc:
                log(f"Warning: could not save W&B run id early: {_exc}")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log(f"Trainer metrics at step {state.global_step}: {json.dumps(logs, sort_keys=True)}")
        return control

    def on_save(self, args, state, control, **kwargs):
        log(f"Trainer callback: checkpoint save triggered at step {state.global_step}")
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log(f"Trainer evaluation at step {state.global_step}: {json.dumps(metrics, sort_keys=True)}")
        else:
            log(f"Trainer evaluation completed at step {state.global_step}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        log(f"Trainer callback: training loop ended at step {state.global_step}")
        return control

# ---------------------------------------------------------------------------
# SIGTERM handling for SLURM wall-time limits
# ---------------------------------------------------------------------------

class GracefulKiller:
    """Catches SIGTERM/SIGUSR1 from SLURM to trigger graceful checkpoint save."""

    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        signal.signal(signal.SIGUSR1, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        log(f"[GracefulKiller] Received signal {signum}. "
            "Will save checkpoint and exit after current step...")
        self.kill_now = True


class SLURMCheckpointCallback(TrainerCallback):
    """Trainer callback that triggers a save + stop when SIGTERM is received."""

    def __init__(self, killer):
        self.killer = killer

    def on_step_end(self, args, state, control, **kwargs):
        if self.killer.kill_now:
            log(f"[SLURMCheckpointCallback] Triggering checkpoint save at step {state.global_step}...")
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Data formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a classifier that determines whether a social media post was "
    "written by a human or generated by AI. Respond with exactly one word: "
    "'ai' or 'human'."
)

# The fixed preamble in every RAG instruction from build_training_data.py
INSTRUCTION_PREAMBLE = (
    "You are a classifier that determines whether a social media post was "
    "written by a human or generated by AI.\n\n"
)


def format_sample(sample, tokenizer):
    """Convert a single {"instruction", "output"} sample to Llama 3.1 chat format.

    Splits the instruction into system + user messages, adds the assistant response.
    Returns the fully formatted text string.
    """
    instruction = sample["instruction"]

    # Extract the user-facing content (RAG examples + post) by stripping the preamble
    if instruction.startswith(INSTRUCTION_PREAMBLE):
        user_content = instruction[len(INSTRUCTION_PREAMBLE):]
    else:
        # Fallback: use everything after the first double-newline
        parts = instruction.split("\n\n", 1)
        user_content = parts[1] if len(parts) > 1 else instruction

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sample["output"]},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def formatting_func(examples, tokenizer):
    """Batch formatting function for SFTTrainer."""
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        sample = {"instruction": instruction, "output": output}
        texts.append(format_sample(sample, tokenizer))
    return {"text": texts}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_verbose_logging()
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version.splitlines()[0]}")

    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B Instruct with QLoRA + RAG data"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpointing (for QoS evidence runs)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log(f"Loaded config from {args.config}")
    log(f"Mode: {'no-checkpoint' if args.no_checkpoint else 'checkpoint'}")
    if args.resume:
        log("Resuming from latest checkpoint")

    # ── W&B setup ──────────────────────────────────────
    wandb_entity = config.get("wandb_entity")
    wandb_project = config.get("wandb_project", "llama-rag-finetune")
    base_output_dir = config.get("output_dir", "./checkpoints/llama-rag")

    # For fresh runs, create a timestamped subdirectory so each run is isolated.
    # For resume runs, find the most recent timestamped subdirectory to continue.
    if args.resume or args.no_checkpoint:
        # Resume: find latest timestamped subdir, fall back to base dir
        import glob as _glob
        subdirs = sorted(_glob.glob(os.path.join(base_output_dir, "run_*")))
        if subdirs:
            output_dir = subdirs[-1]
            log(f"Resuming from existing run dir: {output_dir}")
        else:
            output_dir = base_output_dir
            log(f"No timestamped run dirs found, using base dir: {output_dir}")
    else:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"run_{run_timestamp}")
        log(f"Fresh run: output dir = {output_dir}")

    wandb_run_id_file = os.path.join(output_dir, "wandb_run_id.txt")

    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_MODE"] = "offline"  # Log locally, sync manually after job
    report_to = "wandb"

    try:
        import wandb

        if args.resume and not args.no_checkpoint:
            if os.path.exists(wandb_run_id_file):
                with open(wandb_run_id_file) as _f:
                    saved_run_id = _f.read().strip()
                log(f"W&B resume: restoring run id={saved_run_id}")
                wandb.init(
                    id=saved_run_id,
                    resume="must",
                    entity=wandb_entity,
                    project=wandb_project,
                )
            else:
                log("W&B resume: wandb_run_id.txt not found, starting new run")

        log(f"W&B: entity={wandb_entity}, project={wandb_project}")
    except ImportError:
        log("W&B not installed, falling back to tensorboard")
        report_to = "tensorboard"

    log(
        "Runtime env: "
        f"HF_HUB_VERBOSITY={os.environ.get('HF_HUB_VERBOSITY')} "
        f"HF_DEBUG={os.environ.get('HF_DEBUG')} "
        f"TRANSFORMERS_VERBOSITY={os.environ.get('TRANSFORMERS_VERBOSITY')} "
        f"DATASETS_VERBOSITY={os.environ.get('DATASETS_VERBOSITY')}"
    )

    # ── SIGTERM handler (checkpoint mode only) ─────────
    killer = None
    if not args.no_checkpoint:
        killer = GracefulKiller()
        log("SIGTERM handler registered for graceful checkpoint saves")

    # ── Load model ─────────────────────────────────────
    from unsloth import FastLanguageModel

    model_name = config["model_name"]
    max_seq_length = config.get("max_seq_length", 512)
    load_in_4bit = config.get("load_in_4bit", True)

    log(f"Loading model: {model_name}")
    log(f"  max_seq_length: {max_seq_length}")
    log(f"  load_in_4bit: {load_in_4bit}")
    model_load_start = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )
    log(f"Model and tokenizer loaded in {time.time() - model_load_start:.1f}s")

    # ── Apply LoRA ─────────────────────────────────────
    lora_rank = config.get("lora_rank", 32)
    lora_alpha = config.get("lora_alpha", 64)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    log(f"Applying LoRA: rank={lora_rank}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}")
    log(f"  target_modules: {target_modules}")
    lora_start = time.time()

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.get("seed", 42),
    )
    log(f"LoRA adapters applied in {time.time() - lora_start:.1f}s")

    # ── Load and format dataset ────────────────────────
    train_file = config["train_file"]
    val_file = config.get("val_file")
    max_val_samples = config.get("max_val_samples", 5000)
    seed = config.get("seed", 42)

    log(f"Loading training data from: {train_file}")
    data_files = {"train": train_file}
    if val_file:
        data_files["validation"] = val_file

    dataset_load_start = time.time()
    dataset = load_dataset("json", data_files=data_files)
    log(f"Dataset loaded in {time.time() - dataset_load_start:.1f}s")

    # Subsample validation set for speed
    if val_file and max_val_samples and len(dataset["validation"]) > max_val_samples:
        dataset["validation"] = (
            dataset["validation"]
            .shuffle(seed=seed)
            .select(range(max_val_samples))
        )
        log(f"  Subsampled validation to {max_val_samples} records")

    log(f"  Train: {len(dataset['train'])} records")
    if val_file:
        log(f"  Val: {len(dataset['validation'])} records")

    # Apply chat template formatting
    log("Formatting data to Llama 3.1 chat template...")
    formatting_start = time.time()
    dataset = dataset.map(
        lambda examples: formatting_func(examples, tokenizer),
        batched=True,
        num_proc=4,
        desc="Formatting",
    )
    log(f"Dataset formatting completed in {time.time() - formatting_start:.1f}s")

    # ── Configure training ─────────────────────────────
    from trl import SFTTrainer, SFTConfig

    output_dir = config.get("output_dir", "./checkpoints/llama-rag")

    # Build training arguments
    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
        "learning_rate": config.get("learning_rate", 2e-4),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "lr_scheduler_type": config.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": config.get("warmup_ratio", 0.05),
        "weight_decay": config.get("weight_decay", 0.01),
        "max_grad_norm": config.get("max_grad_norm", 1.0),
        "optim": config.get("optim", "adamw_8bit"),
        "logging_steps": config.get("logging_steps", 50),
        "logging_first_step": True,
        "seed": seed,
        "bf16": True,
        "fp16": False,
        "max_seq_length": max_seq_length,
        "dataset_text_field": "text",
        "report_to": report_to,
        "run_name": f"llama-rag-{time.strftime('%Y%m%d-%H%M%S')}",
        "disable_tqdm": False,
    }

    # Checkpoint vs no-checkpoint mode
    if args.no_checkpoint:
        training_kwargs["save_strategy"] = "no"
        training_kwargs["eval_strategy"] = "no"
        log("Checkpointing DISABLED (QoS evidence mode)")
    else:
        training_kwargs["save_strategy"] = config.get("save_strategy", "steps")
        training_kwargs["save_steps"] = config.get("save_steps", 500)
        training_kwargs["save_total_limit"] = config.get("save_total_limit", 3)
        if val_file:
            training_kwargs["eval_strategy"] = config.get("eval_strategy", "steps")
            training_kwargs["eval_steps"] = config.get("eval_steps", 2000)
        log(f"Checkpointing ENABLED: every {training_kwargs['save_steps']} steps, "
            f"keeping {training_kwargs['save_total_limit']} most recent")

    sft_config = SFTConfig(**training_kwargs)

    # Build callbacks
    callbacks = []
    callbacks.append(VerboseLoggingCallback(
        wandb_run_id_file=wandb_run_id_file if not args.no_checkpoint else None,
        is_resume=args.resume,
    ))
    if killer is not None:
        callbacks.append(SLURMCheckpointCallback(killer))

    # Build trainer
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": dataset["train"],
        "args": sft_config,
        "callbacks": callbacks,
    }
    if val_file and not args.no_checkpoint:
        trainer_kwargs["eval_dataset"] = dataset["validation"]

    log("Building SFTTrainer...")
    trainer_build_start = time.time()
    trainer = SFTTrainer(**trainer_kwargs)
    log(f"SFTTrainer built in {time.time() - trainer_build_start:.1f}s")

    # Restrict loss computation to assistant responses only
    from unsloth import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    log("Loss restricted to assistant responses only")

    # ── Print training summary ─────────────────────────
    total_steps = (
        len(dataset["train"])
        // (training_kwargs["per_device_train_batch_size"]
             * training_kwargs["gradient_accumulation_steps"])
        * config.get("num_train_epochs", 1)
    )
    log("=" * 60)
    log("Training Summary")
    log("=" * 60)
    log(f"  Model:            {model_name}")
    log(f"  LoRA rank:        {lora_rank}")
    log(f"  Train samples:    {len(dataset['train'])}")
    log(f"  Batch size:       {training_kwargs['per_device_train_batch_size']}")
    log(f"  Grad accum:       {training_kwargs['gradient_accumulation_steps']}")
    log(f"  Effective batch:  {training_kwargs['per_device_train_batch_size'] * training_kwargs['gradient_accumulation_steps']}")
    log(f"  Epochs:           {config.get('num_train_epochs', 1)}")
    log(f"  Est. total steps: {total_steps}")
    log(f"  Learning rate:    {training_kwargs['learning_rate']}")
    log(f"  Output dir:       {output_dir}")
    log(f"  W&B:              {report_to}")
    log("=" * 60)

    # ── Train ──────────────────────────────────────────
    log("Starting training...")
    start_time = time.time()

    if args.resume:
        log("Resuming from latest checkpoint...")
        trainer_stats = trainer.train(resume_from_checkpoint=True)
    else:
        trainer_stats = trainer.train()
        # Persist W&B run ID so resume_train.sh can continue the same graph
        if not args.no_checkpoint and report_to == "wandb":
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    with open(wandb_run_id_file, "w") as _f:
                        _f.write(_wandb.run.id)
                    log(f"W&B run id saved: {_wandb.run.id} -> {wandb_run_id_file}")
            except Exception as _exc:
                log(f"Warning: could not save W&B run id: {_exc}")

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    log(f"Training completed in {hours}h {minutes}m {seconds}s")

    # ── Save final model ───────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    log(f"Saving final LoRA adapter to: {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    metrics = {
        "train_runtime_seconds": elapsed,
        "train_runtime_formatted": f"{hours}h {minutes}m {seconds}s",
        "train_samples": len(dataset["train"]),
        "model_name": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": training_kwargs["learning_rate"],
        "effective_batch_size": (
            training_kwargs["per_device_train_batch_size"]
            * training_kwargs["gradient_accumulation_steps"]
        ),
        "num_train_epochs": config.get("num_train_epochs", 1),
        "global_step": trainer.state.global_step,
        "train_loss": trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved training metrics to: {metrics_path}")

    log("Done!")


if __name__ == "__main__":
    main()

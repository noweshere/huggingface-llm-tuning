#!/usr/bin/env python3
"""
================================================================================
 Jetson Orin Nano — Universal LLM LoRA Fine-Tuning Script
================================================================================

 A production-ready, single-file script for fine-tuning any Hugging Face
 causal language model using LoRA (Low-Rank Adaptation) on an NVIDIA Jetson
 Orin Nano (8 GB) or similar resource-constrained GPU.

 Compatible with:
   - PyTorch        2.1.0
   - Transformers   4.38.2
   - NumPy          1.26.4
   - TensorBoard    2.21.0
   - CUDA           12.6
   - PEFT           0.9.x – 0.11.x
   - Accelerate     0.27.x – 0.29.x

 Author : Me
 License: MIT
================================================================================
"""

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — CONFIGURATION                                                 ║
# ║  Edit the variables below to customise your fine-tuning run.               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ── Run Mode Configuration ──────────────────────────────────────────────────
# Mode of operation:
#   "train"     - Run fine-tuning on the dataset.
#   "inference" - Run inference (text generation) on a fine-tuned model or the base model.
RUN_MODE: str = "train"

# --- Inference settings (only used if RUN_MODE is "inference") ---
# The prompt to generate text for.
# If set to "interactive", the script will run an interactive loop in the console
# allowing you to type prompts and receive responses in real-time.
INFERENCE_PROMPT: str = "interactive"

# Path to the model or checkpoint to load for inference.
# If set to None, will auto-detect from OUTPUT_DIR:
#   1. Looks for a merged model under OUTPUT_DIR/merged
#   2. Looks for adapter weights under OUTPUT_DIR/adapter (loads base MODEL_NAME + adapter)
#   3. Falls back to base MODEL_NAME if neither is found.
INFERENCE_MODEL_PATH: str | None = None

# Generation control parameters:
INFERENCE_MAX_NEW_TOKENS: int = 256
INFERENCE_TEMPERATURE: float = 0.7
INFERENCE_TOP_P: float = 0.9
INFERENCE_REPETITION_PENALTY: float = 1.1
INFERENCE_DO_SAMPLE: bool = True


# ── Model Configuration ─────────────────────────────────────────────────────
# The Hugging Face model ID or local path to a causal language model.
# Examples: "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2-0.5B",
#           "google/gemma-2b", "microsoft/phi-2", "mistralai/Mistral-7B-v0.1"
MODEL_NAME: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Tokenizer name — set to None to use the same as MODEL_NAME.
TOKENIZER_NAME = None

# Whether to trust remote code (required for some models like Qwen, Phi, etc.).
TRUST_REMOTE_CODE: bool = True

# Whether to use the fast (Rust-backed) tokenizer when available.
USE_FAST_TOKENIZER: bool = True


# ── Dataset Configuration ───────────────────────────────────────────────────
# Dataset source: "huggingface" for HF Hub datasets, "local" for local files.
DATASET_SOURCE: str = "huggingface"

# --- Hugging Face dataset settings ---
# The HF dataset name (e.g. "wikitext", "tatsu-lab/alpaca", "Open-Orca/OpenOrca").
DATASET_NAME: str = "maharshipandya/spotify-tracks-dataset"

# Optional dataset configuration/subset name (e.g. "wikitext-2-raw-v1").
DATASET_CONFIG = "default"

# --- Local dataset settings ---
# Path to local dataset file or directory.  Supports: .json, .jsonl, .txt, .csv, .parquet
DATASET_PATH = None

# Which split to use for training.  Usually "train".
DATASET_SPLIT: str = "train"

# Percentage of training data to hold out for validation (0–50).
# Only used when the dataset does NOT already have a validation split.
VALIDATION_SPLIT_PERCENT: float = 5.0

# Dataset format hint.  Set to "auto" for automatic detection.
# Options: "auto", "text", "instruction", "chat", "conversation"
DATASET_FORMAT: str = "auto"

# Column names — set to None to auto-detect.
TEXT_FIELD = "artists"        # Column containing raw text (for "text" format)
PROMPT_FIELD = "track_name"      # Column containing the prompt/instruction
RESPONSE_FIELD = "track_genre"    # Column containing the response/output
CONVERSATION_FIELD = "popularity"  # Column containing conversation messages list

# Whether to stream the dataset (useful for very large datasets).
STREAMING: bool = False

# Local directory for caching downloaded datasets.
CACHE_DIR = None


# ── Training Hyperparameters ────────────────────────────────────────────────
# Number of full passes over the training dataset.
EPOCHS: int = 2

# Peak learning rate for the optimiser.
LEARNING_RATE: float = 2e-4

# Micro-batch size per device.  On Jetson 8 GB, keep this at 1 for safety.
BATCH_SIZE: int = 1

# Number of micro-batches to accumulate before a weight update.
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS.
GRADIENT_ACCUMULATION_STEPS: int = 8

# Maximum token length for each training example.  Lower = less VRAM.
# 128–256 recommended for Jetson 8 GB;  512 may work for small models.
MAX_SEQ_LENGTH: int = 256

# Learning-rate scheduler type.
# Options: "linear", "cosine", "cosine_with_restarts", "polynomial",
#          "constant", "constant_with_warmup", "inverse_sqrt"
LR_SCHEDULER: str = "cosine"

# Fraction of total training steps used for learning-rate warmup.
WARMUP_RATIO: float = 0.05

# Weight decay (L2 regularisation) coefficient.
WEIGHT_DECAY: float = 0.01

# Maximum gradient norm for gradient clipping.
MAX_GRAD_NORM: float = 1.0

# Log training metrics every N steps.
LOGGING_STEPS: int = 10

# Save a checkpoint every N steps.  Set to 0 to save only at end of each epoch.
SAVE_STEPS: int = 1000

# Run evaluation every N steps.  Set to 0 to evaluate only at end of each epoch.
EVAL_STEPS: int = 1000

# Random seed for reproducibility.
SEED: int = 42


# ── LoRA Configuration ──────────────────────────────────────────────────────
# LoRA rank (r) — lower = fewer trainable params, higher = more capacity.
# Typical range: 4–64.  8–16 is a good starting point for Jetson.
LORA_RANK: int = 16

# LoRA scaling factor (alpha).  Common rule: alpha = 2 * rank.
LORA_ALPHA: int = 32

# Dropout probability for LoRA layers.
LORA_DROPOUT: float = 0.00

# LoRA bias mode: "none", "all", or "lora_only".
LORA_BIAS: str = "none"

# PEFT task type.  Use "CAUSAL_LM" for autoregressive language models.
LORA_TASK_TYPE: str = "CAUSAL_LM"

# Target modules for LoRA injection.
# Set to "auto" to automatically detect the right modules for the model.
# Or provide a list like: ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_TARGET_MODULES = "auto"


# ── Runtime Configuration ───────────────────────────────────────────────────
# Mixed precision mode.  "fp16" is recommended for Jetson Orin Nano.
# Options: "fp16", "bf16", "no"
MIXED_PRECISION: str = "fp16"

# Enable gradient checkpointing to trade compute for memory.
# Essential on Jetson 8 GB — keep this True.
GRADIENT_CHECKPOINTING: bool = False

# Number of DataLoader worker processes.  0 = load in main process.
# On Jetson with limited RAM, 0 or 1 is safest.
NUM_WORKERS: int = 2

# Output directory for checkpoints, adapters, and merged models.
OUTPUT_DIR: str = "./finetuned_output"

# Path to a checkpoint to resume training from.  Set to None to start fresh.
# Use "latest" to auto-detect the most recent checkpoint in OUTPUT_DIR.
RESUME_CHECKPOINT = "latest"

# Automatically estimate a safe micro-batch size based on available VRAM.
# If True, overrides BATCH_SIZE with the estimated value.
AUTO_BATCH_SIZE: bool = False

# What to save after training:
SAVE_ADAPTERS: bool = True       # Save LoRA adapter weights separately
SAVE_MERGED_MODEL: bool = True  # Save fully merged (base + LoRA) model

# Maximum number of checkpoints to keep (oldest are deleted).
SAVE_TOTAL_LIMIT: int = 3


# ── Optional QLoRA Configuration (DISABLED by default) ──────────────────────
# QLoRA uses 4-bit quantisation via bitsandbytes to dramatically reduce VRAM.
# WARNING: bitsandbytes ARM64/Jetson wheels can be unreliable.  Only enable
# this if you have successfully installed bitsandbytes on your Jetson.
#
# Install (at your own risk):
#   pip install bitsandbytes
#
# Set ENABLE_QLORA = True to activate.
ENABLE_QLORA: bool = False

# Quantisation type for QLoRA.  "nf4" is recommended.
QLORA_QUANT_TYPE: str = "nf4"

# Whether to use double quantisation (saves a bit more memory).
QLORA_DOUBLE_QUANT: bool = True

# Compute dtype for QLoRA operations.
QLORA_COMPUTE_DTYPE: str = "float16"


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — IMPORTS & ENVIRONMENT SETUP                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import os
import sys
import gc
import json
import glob
import math
import time
import random
import logging
import pathlib
import traceback

# ---------------------------------------------------------------------------
# Environment variables — set BEFORE importing PyTorch / CUDA libraries.
# ---------------------------------------------------------------------------
# Reduce VRAM fragmentation on Jetson by enabling expandable memory segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Disable tokenizers parallelism warning when using DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Core imports with version validation.
# ---------------------------------------------------------------------------
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.  Please install PyTorch >= 2.1.0.")
    sys.exit(1)

try:
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        set_seed,
    )
except ImportError:
    print("ERROR: Transformers is not installed.  Please install transformers >= 4.35.")
    sys.exit(1)

try:
    import datasets as hf_datasets
    from datasets import load_dataset, DatasetDict, Dataset
except ImportError:
    print("ERROR: Datasets library is not installed.  Install with: pip install datasets")
    sys.exit(1)

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
    )
except ImportError:
    print(
        "ERROR: PEFT is not installed.  Install with:\n"
        "  pip install 'peft>=0.9,<0.12'"
    )
    sys.exit(1)

# Optional: bitsandbytes for QLoRA (only imported when ENABLE_QLORA is True).
BNB_AVAILABLE = False
prepare_model_for_kbit_training = None  # Will be set if available.

if ENABLE_QLORA:
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        # peft's prepare_model_for_kbit_training may or may not exist depending
        # on peft version — we handle the ImportError gracefully.
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            prepare_model_for_kbit_training = None
        BNB_AVAILABLE = True
    except ImportError:
        print(
            "WARNING: ENABLE_QLORA is True but bitsandbytes is not installed.\n"
            "         Falling back to standard LoRA (no quantisation).\n"
            "         To use QLoRA, install bitsandbytes:  pip install bitsandbytes"
        )
        ENABLE_QLORA = False

# ---------------------------------------------------------------------------
# Version checks — fail fast if something is too old.
# ---------------------------------------------------------------------------
def _parse_version(version_str):
    """Parse a version string like '2.1.0' into a comparable tuple (2, 1, 0)."""
    parts = []
    for part in version_str.split("+")[0].split(".")[:3]:
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


_torch_ver = _parse_version(torch.__version__)
_tf_ver = _parse_version(transformers.__version__)

if _torch_ver < (2, 1, 0):
    print("ERROR: PyTorch {} is too old.  Need >= 2.1.0.".format(torch.__version__))
    sys.exit(1)

if _tf_ver < (4, 35, 0):
    print("ERROR: Transformers {} is too old.  Need >= 4.35.0.".format(
        transformers.__version__))
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging setup.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jetson_finetune")

# Suppress noisy library loggers.
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("peft").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)

logger.info("=" * 72)
logger.info("  Jetson Orin Nano — Universal LLM LoRA Fine-Tuning Script")
logger.info("=" * 72)
logger.info("  PyTorch       : %s", torch.__version__)
logger.info("  Transformers  : %s", transformers.__version__)
logger.info("  Datasets      : %s", hf_datasets.__version__)
logger.info("  NumPy         : %s", np.__version__)
logger.info("  CUDA available: %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("  CUDA version  : %s", torch.version.cuda)
    logger.info("  GPU           : %s", torch.cuda.get_device_name(0))
    _total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info("  GPU memory    : %.2f GB", _total_mem)
logger.info("=" * 72)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — UTILITY FUNCTIONS                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def cleanup_memory():
    """Aggressively free Python and CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    logger.debug("Memory cleanup completed.")


def get_device():
    """
    Return the best available device.
    Defaults to CUDA if available, otherwise falls back to CPU with a warning.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA is not available — falling back to CPU.  "
            "Training will be extremely slow."
        )
    return device


def log_gpu_memory(tag=""):
    """Log current GPU memory usage.  No-op if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved
    prefix = "[{}] ".format(tag) if tag else ""
    logger.info(
        "%sGPU memory — allocated: %.2f GB, reserved: %.2f GB, "
        "free: %.2f GB, total: %.2f GB",
        prefix, allocated, reserved, free, total,
    )


def detect_model_type(config):
    """
    Return a canonical model family string from a Hugging Face model config.
    Used to select LoRA target modules and apply architecture-specific tweaks.
    """
    model_type = getattr(config, "model_type", "unknown").lower()

    # Normalise common aliases.
    FAMILY_MAP = {
        "gpt2": "gpt2",
        "gpt_neo": "gpt_neo",
        "gpt_neox": "gpt_neox",
        "gptj": "gptj",
        "llama": "llama",
        "mistral": "mistral",
        "mixtral": "mistral",
        "qwen": "qwen",
        "qwen2": "qwen2",
        "qwen2_moe": "qwen2",
        "gemma": "gemma",
        "gemma2": "gemma",
        "phi": "phi",
        "phi3": "phi",
        "phimoe": "phi",
        "opt": "opt",
        "bloom": "bloom",
        "falcon": "falcon",
        "mpt": "mpt",
        "stablelm": "stablelm",
        "starcoder2": "starcoder2",
        "cohere": "cohere",
        "internlm": "internlm",
        "internlm2": "internlm",
        "baichuan": "baichuan",
        "chatglm": "chatglm",
    }
    family = FAMILY_MAP.get(model_type, model_type)
    logger.info("Detected model type: '%s' -> family: '%s'", model_type, family)
    return family


def find_all_linear_modules(model):
    """
    Walk the model graph and collect the short names of all nn.Linear layers,
    excluding common output heads that should NOT be adapted.

    Returns a deduplicated list of module short-names (e.g. ["q_proj", "v_proj"]).
    """
    EXCLUDE = {"lm_head", "embed_tokens", "wte", "wpe", "embed_out", "embed_in"}
    linear_names = set()

    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Take the last component of the dotted name.
            short_name = full_name.split(".")[-1]
            if short_name not in EXCLUDE:
                linear_names.add(short_name)

    result = sorted(linear_names)
    logger.info("Found %d unique linear module names: %s", len(result), result)
    return result


def get_default_lora_targets(model_type, model):
    """
    Return a curated list of LoRA target module names for known architectures.
    Falls back to auto-detecting all linear modules for unknown architectures.
    """
    # Curated defaults per model family.
    # These target the attention projections + MLP layers for best quality.
    DEFAULTS = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
        "gemma": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
        "qwen": ["c_attn", "c_proj", "w1", "w2"],
        "phi": ["q_proj", "k_proj", "v_proj", "dense",
                 "fc1", "fc2"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "gpt_neo": ["q_proj", "k_proj", "v_proj", "out_proj",
                     "c_fc", "c_proj"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h",
                      "dense_4h_to_h"],
        "gptj": ["q_proj", "k_proj", "v_proj", "out_proj",
                  "fc_in", "fc_out"],
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj",
                 "fc1", "fc2"],
        "bloom": ["query_key_value", "dense", "dense_h_to_4h",
                   "dense_4h_to_h"],
        "falcon": ["query_key_value", "dense", "dense_h_to_4h",
                    "dense_4h_to_h"],
        "mpt": ["Wqkv", "out_proj", "up_proj", "down_proj"],
        "stablelm": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        "starcoder2": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "c_fc", "c_proj"],
        "cohere": ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
        "internlm": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    }

    if model_type in DEFAULTS:
        targets = DEFAULTS[model_type]
        logger.info("Using curated LoRA targets for '%s': %s", model_type, targets)

        # Validate that these module names actually exist in the model.
        existing = find_all_linear_modules(model)
        validated = [t for t in targets if t in existing]
        if not validated:
            logger.warning(
                "None of the curated targets %s were found in the model.  "
                "Falling back to auto-detection.", targets
            )
            return existing if existing else targets
        if len(validated) < len(targets):
            missing = set(targets) - set(validated)
            logger.info("Some curated targets not found (OK): %s", missing)
        return validated

    # Unknown architecture — auto-detect all linear modules.
    logger.info(
        "No curated LoRA targets for model type '%s'.  "
        "Auto-detecting all linear modules.", model_type
    )
    auto_targets = find_all_linear_modules(model)
    if not auto_targets:
        # Last resort fallback — common attention projections.
        auto_targets = ["q_proj", "v_proj"]
        logger.warning(
            "Could not auto-detect linear modules.  "
            "Using minimal fallback: %s", auto_targets
        )
    return auto_targets


def estimate_safe_batch_size(model, seq_len, mixed_precision="fp16"):
    """
    Estimate the largest safe micro-batch size given available VRAM.

    This is a conservative heuristic — it errs on the side of smaller batches
    to avoid OOM on the Jetson's shared 8 GB memory.
    """
    if not torch.cuda.is_available():
        return 1

    # Bytes per parameter.
    if mixed_precision in ("fp16", "bf16"):
        bytes_per_param = 2
    else:
        bytes_per_param = 4

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model weight memory.
    model_mem = total_params * bytes_per_param

    # Optimizer states for trainable params (Adam: ~8 bytes per trainable param).
    optim_mem = trainable_params * 8

    # Gradient memory for trainable params.
    grad_mem = trainable_params * bytes_per_param

    total_fixed = model_mem + optim_mem + grad_mem

    # Available VRAM (leave 512 MB headroom for OS / display / safety).
    total_vram = torch.cuda.get_device_properties(0).total_mem
    headroom = 512 * 1024 * 1024  # 512 MB
    available = max(total_vram - total_fixed - headroom, 0)

    # Activation memory estimate: very rough — ~4 bytes * hidden_size * seq_len
    # per layer per batch element, halved by gradient checkpointing.
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", 768)
    num_layers = getattr(config, "num_hidden_layers", 12)
    activation_per_sample = num_layers * hidden_size * seq_len * 4
    if GRADIENT_CHECKPOINTING:
        # Gradient checkpointing reduces activation memory by ~sqrt(num_layers).
        activation_per_sample = int(activation_per_sample / math.sqrt(num_layers))

    if activation_per_sample <= 0:
        return 1

    estimated_bs = int(available / activation_per_sample)
    # Clamp to [1, 4] — never go above 4 on an 8 GB device.
    estimated_bs = max(1, min(estimated_bs, 4))

    logger.info(
        "Batch size estimation: model=%.2f GB, optim=%.2f GB, grad=%.2f GB, "
        "available=%.2f GB, activation/sample=%.1f MB -> estimated batch size = %d",
        model_mem / 1e9, optim_mem / 1e9, grad_mem / 1e9,
        available / 1e9, activation_per_sample / 1e6, estimated_bs,
    )
    return estimated_bs


def validate_config():
    """
    Validate all configuration variables and fail fast with helpful messages
    if anything is obviously wrong.
    """
    errors = []

    # --- Run Mode ---
    if RUN_MODE not in ("train", "inference"):
        errors.append(
            "RUN_MODE must be 'train' or 'inference', got '{}'.".format(RUN_MODE)
        )

    # --- Model ---
    if not MODEL_NAME or not isinstance(MODEL_NAME, str):
        errors.append("MODEL_NAME must be a non-empty string.")

    if RUN_MODE == "train":
        # --- Dataset ---
        if DATASET_SOURCE not in ("huggingface", "local"):
            errors.append(
                "DATASET_SOURCE must be 'huggingface' or 'local', "
                "got '{}'.".format(DATASET_SOURCE)
            )
        if DATASET_SOURCE == "huggingface" and not DATASET_NAME:
            errors.append("DATASET_NAME is required when DATASET_SOURCE is 'huggingface'.")
        if DATASET_SOURCE == "local" and not DATASET_PATH:
            errors.append("DATASET_PATH is required when DATASET_SOURCE is 'local'.")
        if DATASET_SOURCE == "local" and DATASET_PATH:
            p = pathlib.Path(DATASET_PATH)
            if not p.exists():
                errors.append("DATASET_PATH does not exist: {}".format(DATASET_PATH))

        if not (0 <= VALIDATION_SPLIT_PERCENT <= 50):
            errors.append(
                "VALIDATION_SPLIT_PERCENT must be between 0 and 50, "
                "got {}.".format(VALIDATION_SPLIT_PERCENT)
            )

        if DATASET_FORMAT not in ("auto", "text", "instruction", "chat", "conversation"):
            errors.append(
                "DATASET_FORMAT must be one of: auto, text, instruction, chat, "
                "conversation.  Got '{}'.".format(DATASET_FORMAT)
            )

        # --- Training ---
        if EPOCHS < 1:
            errors.append("EPOCHS must be >= 1, got {}.".format(EPOCHS))
        if LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be > 0, got {}.".format(LEARNING_RATE))
        if BATCH_SIZE < 1:
            errors.append("BATCH_SIZE must be >= 1, got {}.".format(BATCH_SIZE))
        if GRADIENT_ACCUMULATION_STEPS < 1:
            errors.append(
                "GRADIENT_ACCUMULATION_STEPS must be >= 1, "
                "got {}.".format(GRADIENT_ACCUMULATION_STEPS)
            )
        if MAX_SEQ_LENGTH < 16:
            errors.append("MAX_SEQ_LENGTH must be >= 16, got {}.".format(MAX_SEQ_LENGTH))

        valid_schedulers = {
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup", "inverse_sqrt",
        }
        if LR_SCHEDULER not in valid_schedulers:
            errors.append(
                "LR_SCHEDULER must be one of {}, got '{}'.".format(
                    valid_schedulers, LR_SCHEDULER)
            )

        if not (0 <= WARMUP_RATIO <= 1):
            errors.append("WARMUP_RATIO must be between 0 and 1, "
                           "got {}.".format(WARMUP_RATIO))
        if WEIGHT_DECAY < 0:
            errors.append("WEIGHT_DECAY must be >= 0, got {}.".format(WEIGHT_DECAY))
        if MAX_GRAD_NORM <= 0:
            errors.append("MAX_GRAD_NORM must be > 0, got {}.".format(MAX_GRAD_NORM))

        # --- LoRA ---
        if LORA_RANK < 1:
            errors.append("LORA_RANK must be >= 1, got {}.".format(LORA_RANK))
        if LORA_ALPHA < 1:
            errors.append("LORA_ALPHA must be >= 1, got {}.".format(LORA_ALPHA))
        if not (0 <= LORA_DROPOUT < 1):
            errors.append("LORA_DROPOUT must be in [0, 1), got {}.".format(LORA_DROPOUT))
        if LORA_BIAS not in ("none", "all", "lora_only"):
            errors.append(
                "LORA_BIAS must be 'none', 'all', or 'lora_only', "
                "got '{}'.".format(LORA_BIAS)
            )

    elif RUN_MODE == "inference":
        # --- Inference ---
        if not isinstance(INFERENCE_PROMPT, str):
            errors.append("INFERENCE_PROMPT must be a string.")
        if INFERENCE_MAX_NEW_TOKENS < 1:
            errors.append("INFERENCE_MAX_NEW_TOKENS must be >= 1.")
        if INFERENCE_TEMPERATURE < 0.0:
            errors.append("INFERENCE_TEMPERATURE must be >= 0.0.")
        if not (0.0 <= INFERENCE_TOP_P <= 1.0):
            errors.append("INFERENCE_TOP_P must be between 0.0 and 1.0.")
        if INFERENCE_REPETITION_PENALTY <= 0.0:
            errors.append("INFERENCE_REPETITION_PENALTY must be > 0.0.")

    # --- Runtime ---
    if MIXED_PRECISION not in ("fp16", "bf16", "no"):
        errors.append(
            "MIXED_PRECISION must be 'fp16', 'bf16', or 'no', "
            "got '{}'.".format(MIXED_PRECISION)
        )

    # --- Report ---
    if errors:
        logger.error("Configuration validation failed:")
        for i, err in enumerate(errors, 1):
            logger.error("  %d. %s", i, err)
        sys.exit(1)
    else:
        logger.info("Configuration validation passed.")


def find_latest_checkpoint(output_dir):
    """
    Find the most recent checkpoint directory inside output_dir.
    Checkpoints follow the naming convention 'checkpoint-NNNN'.
    Returns the full path, or None if no checkpoints are found.
    """
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None

    # Sort by the step number embedded in the directory name.
    def _step_num(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    checkpoint_dirs.sort(key=_step_num)
    latest = checkpoint_dirs[-1]
    logger.info("Found latest checkpoint: %s", latest)
    return latest


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — TOKENIZER SETUP                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def setup_tokenizer(model_name):
    """
    Load and configure the tokenizer for the specified model.

    Handles:
      - GPT-2 tokenizer detection and left-padding.
      - Missing pad tokens (falls back to EOS).
      - SentencePiece, BPE, and other tokenizer types.
      - trust_remote_code for custom tokenizers.
    """
    tokenizer_name = TOKENIZER_NAME or model_name

    logger.info("Loading tokenizer: %s", tokenizer_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=USE_FAST_TOKENIZER,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.error("Failed to load tokenizer '%s': %s", tokenizer_name, e)
        raise

    # ── Detect tokenizer type ────────────────────────────────────────────────
    tokenizer_class_name = type(tokenizer).__name__
    is_gpt2_tokenizer = "gpt2" in tokenizer_class_name.lower() or (
        hasattr(tokenizer, "name_or_path")
        and "gpt2" in tokenizer.name_or_path.lower()
    )

    logger.info("Tokenizer class : %s", tokenizer_class_name)
    logger.info("Vocab size      : %d", len(tokenizer))
    logger.info("Is GPT-2 style  : %s", is_gpt2_tokenizer)

    # ── Handle missing pad token ─────────────────────────────────────────────
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(
                "Pad token was missing — set to EOS token: '%s' (id=%d)",
                tokenizer.pad_token, tokenizer.pad_token_id,
            )
        else:
            # Extremely rare edge case — add a new pad token.
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.warning(
                "Both pad_token and eos_token were missing.  "
                "Added '[PAD]' as pad_token."
            )

    # ── GPT-2 specific: use left padding for causal LM ──────────────────────
    if is_gpt2_tokenizer:
        tokenizer.padding_side = "left"
        logger.info("Set padding_side='left' for GPT-2 tokenizer.")

    # ── Log special tokens ───────────────────────────────────────────────────
    logger.info(
        "Special tokens — BOS: %r, EOS: %r, PAD: %r",
        tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token,
    )

    return tokenizer


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MODEL LOADING                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def load_model(model_name, device, tokenizer):
    """
    Load the causal language model and return (model, model_type_string).

    Handles:
      - FP16 / BF16 dtype selection.
      - low_cpu_mem_usage to avoid peak 2x memory during loading.
      - Embedding resizing if tokenizer has extra tokens.
      - Gradient checkpointing.
      - Optional QLoRA quantisation.
      - Model compatibility validation.
    """
    logger.info("Loading model: %s", model_name)
    cleanup_memory()
    log_gpu_memory("before model load")

    # ── Load config first to inspect architecture ────────────────────────────
    try:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=TRUST_REMOTE_CODE,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.error("Failed to load model config for '%s': %s", model_name, e)
        raise

    model_type = detect_model_type(config)

    # ── Estimate and log model size ──────────────────────────────────────────
    hidden = getattr(config, "hidden_size", 768)
    layers = getattr(config, "num_hidden_layers", 12)
    vocab = getattr(config, "vocab_size", 50257)
    approx_params = (hidden * hidden * 12 * layers) + (hidden * vocab * 2)
    approx_gb = approx_params * 2 / (1024 ** 3)  # FP16 bytes
    logger.info(
        "Model architecture: %d layers, hidden=%d, vocab=%d  "
        "(~%.2fB params, ~%.1f GB in FP16)",
        layers, hidden, vocab, approx_params / 1e9, approx_gb,
    )

    if approx_gb > 6.0:
        logger.warning(
            "This model is estimated at ~%.1f GB in FP16 — it may be "
            "too large for 8 GB Jetson.  Consider a smaller model or "
            "enabling QLoRA.", approx_gb,
        )

    # ── Determine torch dtype ────────────────────────────────────────────────
    if MIXED_PRECISION == "bf16":
        torch_dtype = torch.bfloat16
    elif MIXED_PRECISION == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # ── Build loading kwargs ─────────────────────────────────────────────────
    load_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=TRUST_REMOTE_CODE,
        cache_dir=CACHE_DIR,
    )

    # ── QLoRA quantisation config (if enabled) ───────────────────────────────
    if ENABLE_QLORA and BNB_AVAILABLE:
        logger.info("QLoRA enabled — loading model with 4-bit quantisation.")
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        qlora_compute_dt = compute_dtype_map.get(QLORA_COMPUTE_DTYPE, torch.float16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QLORA_QUANT_TYPE,
            bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=qlora_compute_dt,
        )
        load_kwargs["quantization_config"] = bnb_config
        # With quantisation, let accelerate handle device placement.
        load_kwargs["device_map"] = "auto"
    else:
        # On single-GPU Jetson without quantisation, manual placement is safest.
        load_kwargs["device_map"] = None

    # ── Load the model ───────────────────────────────────────────────────────
    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    except Exception as e:
        logger.error("Failed to load model '%s': %s", model_name, e)
        logger.error(
            "If this is a gated model, make sure you have accepted the license "
            "and set your HF_TOKEN environment variable."
        )
        raise

    # ── QLoRA: prepare model for k-bit training ─────────────────────────────
    if ENABLE_QLORA and BNB_AVAILABLE:
        if prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
            )
            logger.info("Model prepared for k-bit training.")
        else:
            logger.warning(
                "prepare_model_for_kbit_training not available in this PEFT version.  "
                "Continuing without it — training may be less stable."
            )

    # ── Move to device (non-QLoRA path) ──────────────────────────────────────
    if not (ENABLE_QLORA and BNB_AVAILABLE):
        model = model.to(device)

    log_gpu_memory("after model load")

    # ── Resize embeddings if tokenizer is larger ─────────────────────────────
    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)

    if tokenizer_vocab_size != model_vocab_size:
        logger.info(
            "Resizing model embeddings: %d -> %d",
            model_vocab_size, tokenizer_vocab_size,
        )
        model.resize_token_embeddings(tokenizer_vocab_size)

    # ── Enable gradient checkpointing ────────────────────────────────────────
    if GRADIENT_CHECKPOINTING and not (ENABLE_QLORA and BNB_AVAILABLE):
        # For QLoRA, gradient checkpointing is handled by
        # prepare_model_for_kbit_training above.
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing: %s", e)

    # For gradient checkpointing compatibility with the Trainer,
    # inputs must require gradients.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Fallback for older model implementations.
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(
            _make_inputs_require_grad
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded successfully.  Total params: %.1fM", total_params / 1e6)

    return model, model_type


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — LoRA CONFIGURATION                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def setup_lora(model, model_type):
    """
    Configure and apply LoRA adapters to the model.

    Automatically detects target modules when LORA_TARGET_MODULES == "auto",
    or uses the user-provided list.

    Returns (peft_model, lora_config).
    """
    logger.info("Configuring LoRA adapter...")

    # ── Determine target modules ─────────────────────────────────────────────
    if LORA_TARGET_MODULES == "auto":
        target_modules = get_default_lora_targets(model_type, model)
    elif isinstance(LORA_TARGET_MODULES, str):
        # Single module name as a string — wrap in list.
        target_modules = [LORA_TARGET_MODULES]
    else:
        target_modules = list(LORA_TARGET_MODULES)

    if not target_modules:
        logger.error("No target modules for LoRA — cannot continue.")
        sys.exit(1)

    logger.info("LoRA target modules: %s", target_modules)

    # ── Map task type string to PEFT enum ────────────────────────────────────
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "SEQ_CLS": TaskType.SEQ_CLS,
    }
    task_type = task_type_map.get(LORA_TASK_TYPE.upper(), TaskType.CAUSAL_LM)

    # ── Build LoRA config ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=task_type,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=target_modules,
    )

    logger.info(
        "LoRA config — rank=%d, alpha=%d, dropout=%.3f, bias='%s', task_type=%s",
        LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, task_type,
    )

    # ── Apply LoRA to the model ──────────────────────────────────────────────
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        logger.error("Failed to apply LoRA adapter: %s", e)
        logger.error(
            "This may happen if the target modules don't match the model "
            "architecture.  Try setting LORA_TARGET_MODULES to a specific list "
            "of module names.  You can inspect the model with:  print(model)"
        )
        raise

    # ── Convert only trainable parameters to float32 ──────────────────────────
    # This prevents the "Attempting to unscale FP16 gradients" error under AMP fp16.
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # ── Log trainable parameters ─────────────────────────────────────────────
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        "{:,}".format(trainable), "{:,}".format(total),
        100.0 * trainable / total,
    )

    log_gpu_memory("after LoRA setup")

    return model, lora_config


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — DATASET LOADING & PREPROCESSING                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def load_raw_dataset():
    """
    Load raw dataset from Hugging Face Hub or local files.

    Returns a DatasetDict with at least a 'train' key, and optionally a
    'validation' (or 'test') key.
    """
    logger.info("Loading dataset (source=%s)...", DATASET_SOURCE)

    if DATASET_SOURCE == "huggingface":
        return _load_hf_dataset()
    else:
        return _load_local_dataset()


def _load_hf_dataset():
    """Load a dataset from the Hugging Face Hub."""
    load_kwargs = dict(
        path=DATASET_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if DATASET_CONFIG:
        load_kwargs["name"] = DATASET_CONFIG
    if STREAMING:
        load_kwargs["streaming"] = True

    try:
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", DATASET_NAME, e)
        logger.error(
            "Check that the dataset name is correct, you have internet access, "
            "and if it's a gated dataset, that you have access."
        )
        raise

    # If load_dataset returns a DatasetDict, use it directly.
    if isinstance(dataset, DatasetDict):
        logger.info("Dataset splits: %s", list(dataset.keys()))

        # Ensure we have a training split.
        if "train" not in dataset:
            # Try to use the first available split.
            available = list(dataset.keys())
            if available:
                logger.warning(
                    "No 'train' split found.  Using '%s' as training data.",
                    available[0],
                )
                dataset = DatasetDict({"train": dataset[available[0]]})
            else:
                raise ValueError("Dataset has no splits at all.")

        # Create validation split if missing.
        has_val = any(
            k in dataset
            for k in ("validation", "valid", "val", "test", "dev")
        )
        if not has_val and VALIDATION_SPLIT_PERCENT > 0 and not STREAMING:
            logger.info(
                "No validation split found — creating one "
                "(%.1f%% of training data).", VALIDATION_SPLIT_PERCENT,
            )
            split = dataset["train"].train_test_split(
                test_size=VALIDATION_SPLIT_PERCENT / 100.0,
                seed=SEED,
            )
            dataset = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
            })
        elif has_val:
            # Normalise the validation split key.
            for key in ("validation", "valid", "val", "dev", "test"):
                if key in dataset and key != "validation":
                    dataset = DatasetDict({
                        "train": dataset["train"],
                        "validation": dataset[key],
                    })
                    break

        return dataset

    # If we got a single Dataset (e.g. by specifying a split), wrap it.
    if isinstance(dataset, Dataset):
        ds_dict = DatasetDict({"train": dataset})
        if VALIDATION_SPLIT_PERCENT > 0 and not STREAMING:
            split = dataset.train_test_split(
                test_size=VALIDATION_SPLIT_PERCENT / 100.0, seed=SEED,
            )
            ds_dict = DatasetDict({
                "train": split["train"],
                "validation": split["test"],
            })
        return ds_dict

    # Streaming IterableDataset — wrap without split.
    return DatasetDict({"train": dataset})


def _load_local_dataset():
    """Load a dataset from local files."""
    path = pathlib.Path(DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Dataset path does not exist: {}".format(DATASET_PATH)
        )

    logger.info("Loading local dataset from: %s", path)

    # Determine file type and corresponding loader.
    if path.is_file():
        files = [path]
    else:
        # Directory — find data files.
        supported_exts = {".json", ".jsonl", ".csv", ".parquet", ".txt"}
        files = [
            f for f in path.iterdir() if f.suffix.lower() in supported_exts
        ]
        if not files:
            raise FileNotFoundError(
                "No supported data files found in {}.  "
                "Supported extensions: {}".format(DATASET_PATH, supported_exts)
            )
        logger.info("Found %d data files in directory.", len(files))

    # Use the extension of the first file to determine the format.
    ext = files[0].suffix.lower()
    file_paths = [str(f) for f in files]

    format_map = {
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
        ".parquet": "parquet",
        ".txt": "text",
    }
    loader_format = format_map.get(ext)
    if loader_format is None:
        raise ValueError("Unsupported file extension: {}".format(ext))

    logger.info("Loading as '%s' format.", loader_format)

    try:
        dataset = load_dataset(
            loader_format,
            data_files=file_paths,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.error("Failed to load local dataset: %s", e)
        raise

    # The load_dataset call usually returns a DatasetDict with a 'train' key.
    if isinstance(dataset, DatasetDict) and "train" in dataset:
        ds = dataset
    elif isinstance(dataset, DatasetDict):
        # Use the first available split.
        first_key = list(dataset.keys())[0]
        ds = DatasetDict({"train": dataset[first_key]})
    else:
        ds = DatasetDict({"train": dataset})

    # Create validation split if needed.
    if "validation" not in ds and VALIDATION_SPLIT_PERCENT > 0:
        logger.info(
            "Creating validation split (%.1f%% of training data).",
            VALIDATION_SPLIT_PERCENT,
        )
        split = ds["train"].train_test_split(
            test_size=VALIDATION_SPLIT_PERCENT / 100.0, seed=SEED,
        )
        ds = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })

    return ds


def detect_dataset_format(dataset):
    """
    Auto-detect the dataset format based on column names.

    Returns one of: "text", "instruction", "chat", "conversation".
    """
    columns = set(dataset.column_names)
    logger.info("Dataset columns: %s", sorted(columns))

    # Conversation / chat format.
    conversation_keys = {"conversations", "messages", "dialogue", "dialog"}
    if columns & conversation_keys:
        fmt = "conversation"
        logger.info("Auto-detected dataset format: %s", fmt)
        return fmt

    # Instruction format.
    instruction_keys = {"instruction", "prompt", "question", "input"}
    response_keys = {"output", "response", "answer", "completion", "text"}
    if (columns & instruction_keys) and (columns & response_keys):
        fmt = "instruction"
        logger.info("Auto-detected dataset format: %s", fmt)
        return fmt

    # Plain text format.
    text_keys = {"text", "content", "document", "passage", "sentence"}
    if columns & text_keys:
        fmt = "text"
        logger.info("Auto-detected dataset format: %s", fmt)
        return fmt

    # Fallback: use the first column.
    logger.warning(
        "Could not auto-detect dataset format from columns %s.  "
        "Falling back to 'text' using the first column.", sorted(columns),
    )
    return "text"


def _find_column(dataset, candidates):
    """Return the first column name from candidates that exists in the dataset."""
    columns = set(dataset.column_names)
    for c in candidates:
        if c in columns:
            return c
    return None


def format_examples(examples, fmt, tokenizer):
    """
    Convert raw dataset rows into a single 'text' field per example.

    This function is designed to be used with Dataset.map(batched=True).
    """
    batch_size = len(next(iter(examples.values())))
    texts = []

    for i in range(batch_size):
        if fmt == "text":
            # ── Plain text ───────────────────────────────────────────────
            field = TEXT_FIELD or _find_text_field_name(examples)
            text = str(examples[field][i]).strip()

        elif fmt == "instruction":
            # ── Instruction / prompt-response ────────────────────────────
            prompt_field = PROMPT_FIELD or _find_instruction_field_name(examples)
            response_field = RESPONSE_FIELD or _find_response_field_name(examples)

            prompt = str(examples[prompt_field][i]).strip()
            response = str(examples[response_field][i]).strip()

            # Optional input/context field.
            input_field = _find_column_in_batch(examples, ["input", "context"])
            if input_field and examples[input_field][i]:
                input_text = str(examples[input_field][i]).strip()
                text = (
                    "### Instruction:\n{}\n\n"
                    "### Input:\n{}\n\n"
                    "### Response:\n{}".format(prompt, input_text, response)
                )
            else:
                text = (
                    "### Instruction:\n{}\n\n"
                    "### Response:\n{}".format(prompt, response)
                )

        elif fmt in ("chat", "conversation"):
            # ── Conversation / chat ──────────────────────────────────────
            conv_field = (CONVERSATION_FIELD
                          or _find_conversation_field_name(examples))
            messages = examples[conv_field][i]

            # Try tokenizer's chat template first.
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    # Ensure messages are in the right format.
                    if isinstance(messages, str):
                        messages = json.loads(messages)
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                    )
                except Exception:
                    # Fallback to manual formatting.
                    text = _format_conversation_manual(messages)
            else:
                text = _format_conversation_manual(messages)
        else:
            # Shouldn't happen — treat as text.
            field = TEXT_FIELD or _find_text_field_name(examples)
            text = str(examples[field][i]).strip()

        # Append EOS token to mark the end of the training example.
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token

        texts.append(text)

    return {"text": texts}


def _find_text_field_name(examples):
    """Find the text column in a batch of examples."""
    for candidate in ["text", "content", "document", "passage", "sentence"]:
        if candidate in examples:
            return candidate
    # Fallback: first key.
    return next(iter(examples.keys()))


def _find_instruction_field_name(examples):
    """Find the instruction/prompt column."""
    for candidate in ["instruction", "prompt", "question", "input_text"]:
        if candidate in examples:
            return candidate
    return next(iter(examples.keys()))


def _find_response_field_name(examples):
    """Find the response/output column."""
    for candidate in ["output", "response", "answer", "completion",
                       "target", "target_text"]:
        if candidate in examples:
            return candidate
    keys = list(examples.keys())
    return keys[1] if len(keys) > 1 else keys[0]


def _find_conversation_field_name(examples):
    """Find the conversation/messages column."""
    for candidate in ["conversations", "messages", "dialogue", "dialog", "chat"]:
        if candidate in examples:
            return candidate
    return next(iter(examples.keys()))


def _find_column_in_batch(examples, candidates):
    """Find the first matching column name in a batch."""
    for c in candidates:
        if c in examples:
            return c
    return None


def _format_conversation_manual(messages):
    """
    Manually format a conversation into a text string when no chat template
    is available.

    Handles both list-of-dicts and string representations.
    """
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return messages  # Return as-is if not valid JSON.

    if not isinstance(messages, list):
        return str(messages)

    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            # Standard format: {"role": "user", "content": "..."}
            role = msg.get("role", msg.get("from", "unknown"))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
            role_label = role.capitalize()
            parts.append("{}: {}".format(role_label, content))
        elif isinstance(msg, str):
            parts.append(msg)
        else:
            parts.append(str(msg))

    return "\n\n".join(parts)


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize all splits in the dataset.

    After tokenization, each example has 'input_ids', 'attention_mask', and
    'labels' fields ready for causal language modelling.
    """
    logger.info("Tokenizing dataset...")

    def _tokenize_fn(examples):
        """Tokenize a batch of text examples."""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # Dynamic padding via DataCollator at training time.
            return_attention_mask=True,
        )
        # For causal LM, labels are the same as input_ids.
        # The Trainer / DataCollator handles shifting internally.
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    # Remove all columns except the tokenized ones.
    tokenized_dataset = DatasetDict()
    for split_name, split_data in dataset.items():
        logger.info(
            "  Tokenizing '%s' split (%d examples)...",
            split_name, len(split_data),
        )

        # Filter out empty texts before tokenizing.
        original_size = len(split_data)
        split_data = split_data.filter(
            lambda x: x.get("text", "") and len(x["text"].strip()) > 0,
            desc="Filtering empty texts ({})".format(split_name),
        )
        filtered_size = len(split_data)
        if filtered_size < original_size:
            logger.info(
                "  Filtered %d empty examples from '%s'.",
                original_size - filtered_size, split_name,
            )

        tokenized_split = split_data.map(
            _tokenize_fn,
            batched=True,
            batch_size=1000,
            remove_columns=split_data.column_names,
            desc="Tokenizing {}".format(split_name),
            num_proc=1,  # Single-process on Jetson to save RAM.
        )
        tokenized_dataset[split_name] = tokenized_split
        logger.info(
            "  '%s': %d tokenized examples.", split_name, len(tokenized_split),
        )

    return tokenized_dataset


def prepare_dataset(tokenizer):
    """
    Full pipeline: load -> detect format -> format -> tokenize.
    """
    # Step 1: Load raw dataset.
    raw_dataset = load_raw_dataset()

    # Step 2: Detect format.
    train_data = raw_dataset["train"]
    if DATASET_FORMAT == "auto":
        fmt = detect_dataset_format(train_data)
    else:
        fmt = DATASET_FORMAT
    logger.info("Using dataset format: %s", fmt)

    # Step 3: Format examples into a unified 'text' field.
    logger.info("Formatting dataset examples...")
    formatted_dataset = DatasetDict()
    for split_name, split_data in raw_dataset.items():
        formatted = split_data.map(
            lambda examples: format_examples(examples, fmt, tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=split_data.column_names,
            desc="Formatting {}".format(split_name),
            num_proc=1,
        )
        formatted_dataset[split_name] = formatted

    # Step 4: Tokenize.
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer)

    # Log dataset sizes.
    for split_name, split_data in tokenized_dataset.items():
        logger.info("  Final '%s' dataset: %d examples", split_name, len(split_data))

    return tokenized_dataset


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — TRAINING SETUP                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def build_trainer(model, tokenizer, train_dataset, eval_dataset, batch_size):
    """
    Build the Hugging Face Trainer with optimised settings for Jetson.
    """
    logger.info("Building Trainer...")

    # ── Output and logging directories ───────────────────────────────────────
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    # ── Mixed precision flags ────────────────────────────────────────────────
    use_fp16 = MIXED_PRECISION == "fp16" and torch.cuda.is_available()
    use_bf16 = MIXED_PRECISION == "bf16" and torch.cuda.is_available()

    # ── Evaluation strategy ──────────────────────────────────────────────────
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_strategy = "steps" if EVAL_STEPS > 0 else "epoch"
        eval_steps = EVAL_STEPS if EVAL_STEPS > 0 else None
        load_best = True
    else:
        eval_strategy = "no"
        eval_steps = None
        load_best = False

    # ── Save strategy ────────────────────────────────────────────────────────
    save_strategy = "steps" if SAVE_STEPS > 0 else "epoch"
    save_steps = SAVE_STEPS if SAVE_STEPS > 0 else None

    # ── Build training arguments ─────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,

        # Epochs and batching.
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimiser.
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,

        # Precision.
        fp16=use_fp16,
        bf16=use_bf16,

        # Gradient checkpointing.
        gradient_checkpointing=GRADIENT_CHECKPOINTING,

        # Logging.
        logging_dir=str(logging_dir),
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="tensorboard",

        # Saving.
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,

        # Evaluation.
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,

        # DataLoader.
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=False,  # Jetson unified memory — pinning is a no-op.

        # Reproducibility.
        seed=SEED,
        data_seed=SEED,

        # Miscellaneous.
        remove_unused_columns=False,  # Avoid accidental column drops with PEFT.
        disable_tqdm=False,
    )

    # ── Data collator with dynamic padding ───────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM — no masked language modelling.
    )

    # ── Build Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info(
        "Trainer built — epochs=%d, batch_size=%d, grad_accum=%d, "
        "effective_batch=%d, precision=%s, lr=%s, scheduler=%s",
        EPOCHS, batch_size, GRADIENT_ACCUMULATION_STEPS,
        batch_size * GRADIENT_ACCUMULATION_STEPS,
        "fp16" if use_fp16 else ("bf16" if use_bf16 else "fp32"),
        LEARNING_RATE, LR_SCHEDULER,
    )

    return trainer


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — OOM-SAFE TRAINING LOOP                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def train_with_oom_recovery(trainer):
    """
    Run training with automatic CUDA OOM recovery.

    If an OutOfMemoryError occurs, the function:
      1. Clears GPU memory.
      2. Halves the micro-batch size (minimum 1).
      3. Doubles gradient accumulation to maintain effective batch size.
      4. Retries training (up to 2 retries).
    """
    max_retries = 2
    current_batch_size = trainer.args.per_device_train_batch_size
    current_grad_accum = trainer.args.gradient_accumulation_steps

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    resume_from = None
    if RESUME_CHECKPOINT == "latest":
        resume_from = find_latest_checkpoint(OUTPUT_DIR)
    elif RESUME_CHECKPOINT:
        if os.path.isdir(RESUME_CHECKPOINT):
            resume_from = RESUME_CHECKPOINT
        else:
            logger.warning(
                "Checkpoint path not found: %s.  Starting from scratch.",
                RESUME_CHECKPOINT,
            )

    if resume_from:
        logger.info("Resuming training from checkpoint: %s", resume_from)
    else:
        logger.info("Starting training from scratch.")

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "Training attempt %d/%d — batch_size=%d, grad_accum=%d",
                attempt + 1, max_retries + 1,
                current_batch_size, current_grad_accum,
            )
            log_gpu_memory("before training")

            train_result = trainer.train(resume_from_checkpoint=resume_from)

            # Training succeeded.
            logger.info("Training completed successfully!")
            metrics = train_result.metrics
            logger.info("Training metrics: %s", metrics)

            # Save metrics.
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # Run final evaluation if we have an eval dataset.
            if (trainer.eval_dataset is not None
                    and len(trainer.eval_dataset) > 0):
                logger.info("Running final evaluation...")
                eval_metrics = trainer.evaluate()
                logger.info("Evaluation metrics: %s", eval_metrics)
                trainer.log_metrics("eval", eval_metrics)
                trainer.save_metrics("eval", eval_metrics)

            return  # Success — exit the retry loop.

        except torch.cuda.OutOfMemoryError as oom_error:
            logger.error("CUDA Out-Of-Memory error: %s", oom_error)
            cleanup_memory()
            log_gpu_memory("after OOM cleanup")

            if attempt >= max_retries:
                logger.error(
                    "Max OOM retries exceeded.  Training cannot continue.  "
                    "Try reducing MAX_SEQ_LENGTH, LORA_RANK, or using a "
                    "smaller model."
                )
                raise

            # ── Reduce batch size and increase grad accumulation ─────────
            if current_batch_size > 1:
                new_batch_size = max(1, current_batch_size // 2)
                new_grad_accum = current_grad_accum * (
                    current_batch_size // new_batch_size
                )
                logger.warning(
                    "Reducing batch size: %d -> %d, "
                    "increasing grad_accum: %d -> %d",
                    current_batch_size, new_batch_size,
                    current_grad_accum, new_grad_accum,
                )
                current_batch_size = new_batch_size
                current_grad_accum = new_grad_accum
            else:
                logger.warning(
                    "Batch size is already 1.  Attempting to continue with "
                    "more aggressive memory cleanup."
                )

            # Update trainer arguments.
            trainer.args.per_device_train_batch_size = current_batch_size
            trainer.args.gradient_accumulation_steps = current_grad_accum

            # Don't try to resume from a checkpoint after OOM — start fresh.
            resume_from = None

        except KeyboardInterrupt:
            logger.warning(
                "Training interrupted by user.  Saving emergency checkpoint..."
            )
            try:
                trainer.save_model(
                    os.path.join(OUTPUT_DIR, "emergency_checkpoint")
                )
                trainer.save_state()
                logger.info("Emergency checkpoint saved.")
            except Exception as save_err:
                logger.error(
                    "Failed to save emergency checkpoint: %s", save_err
                )
            raise

        except Exception as e:
            logger.error("Unexpected training error: %s", e)
            logger.error(traceback.format_exc())

            # Try to save an emergency checkpoint.
            try:
                emergency_dir = os.path.join(OUTPUT_DIR, "emergency_checkpoint")
                logger.info(
                    "Saving emergency checkpoint to %s...", emergency_dir
                )
                trainer.save_model(emergency_dir)
                trainer.save_state()
            except Exception:
                logger.error("Could not save emergency checkpoint.")

            raise


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — MODEL SAVING                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def save_results(model, tokenizer):
    """
    Save the trained model according to the configuration:
      - SAVE_ADAPTERS: save LoRA adapter weights separately.
      - SAVE_MERGED_MODEL: merge LoRA into the base model and save.
      - Tokenizer is always saved alongside.
    """
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save tokenizer ───────────────────────────────────────────────────────
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    logger.info("Tokenizer saved to: %s", tokenizer_dir)

    # ── Save LoRA adapters ───────────────────────────────────────────────────
    if SAVE_ADAPTERS:
        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        # Also save tokenizer with adapter for convenience.
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("LoRA adapter saved to: %s", adapter_dir)

        # Log adapter size.
        adapter_size = sum(
            f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()
        )
        logger.info("Adapter size: %.1f MB", adapter_size / (1024 ** 2))

    # ── Save merged model ────────────────────────────────────────────────────
    if SAVE_MERGED_MODEL:
        logger.info("Merging LoRA weights into base model (High-Performance Mode)...")
        cleanup_memory()

        try:
            # 1. Unload model from GPU to CPU to free VRAM completely
            logger.info("Moving model to CPU for memory-efficient merging...")
            model = model.to("cpu")
            cleanup_memory()

            # 2. Merge LoRA adapter weights in half precision on CPU
            # safetensors and torch overhead is minimized
            with torch.no_grad():
                merged_model = model.merge_and_unload(progressbar=True)

            # 3. Define optimized output directory
            merged_dir = output_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)

            # 4. Save using SafeTensors with chunked shards
            logger.info("Saving merged model with SafeTensors and 2GB sharding...")
            merged_model.save_pretrained(
                str(merged_dir),
                safe_serialization=True,     # Fast memory-mapped loading
                max_shard_size="2GB",        # Lowers peak RAM overhead during load/save
            )
            tokenizer.save_pretrained(str(merged_dir))

            # Clean up temporary merged model object
            del merged_model
            cleanup_memory()

            logger.info("Merged model saved successfully to: %s", merged_dir)

        except Exception as e:
            logger.error("Failed to merge model: %s", e)
            if not SAVE_ADAPTERS:
                # Emergency save of the adapter if we haven't already.
                emergency_adapter_dir = output_dir / "adapter_emergency"
                model.save_pretrained(str(emergency_adapter_dir))
                tokenizer.save_pretrained(str(emergency_adapter_dir))
                logger.info(
                    "Emergency adapter saved to: %s", emergency_adapter_dir
                )

    if not SAVE_ADAPTERS and not SAVE_MERGED_MODEL:
        logger.warning(
            "Neither SAVE_ADAPTERS nor SAVE_MERGED_MODEL is True.  "
            "Saving adapter by default to prevent data loss."
        )
        fallback_dir = output_dir / "adapter"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(fallback_dir))
        tokenizer.save_pretrained(str(fallback_dir))
        logger.info("Adapter saved (fallback) to: %s", fallback_dir)

    logger.info("All save operations complete.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10B — INFERENCE GENERATION                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def run_inference():
    """
    Run inference (text generation) on a fine-tuned model or adapter.
    """
    logger.info("=" * 72)
    logger.info("  RUNNING INFERENCE MODE")
    logger.info("=" * 72)

    device = get_device()
    cleanup_memory()

    # 1. Resolve tokenizer and model path
    model_path = INFERENCE_MODEL_PATH
    tokenizer_path = None

    if model_path is None:
        # Try auto-detecting
        output_dir = pathlib.Path(OUTPUT_DIR)
        merged_path = output_dir / "merged"
        adapter_path = output_dir / "adapter"

        if merged_path.exists() and merged_path.is_dir():
            model_path = str(merged_path)
            logger.info("Auto-detected merged model path: %s", model_path)
        elif adapter_path.exists() and adapter_path.is_dir():
            model_path = str(adapter_path)
            logger.info("Auto-detected LoRA adapter path: %s", model_path)
        else:
            model_path = MODEL_NAME
            logger.warning(
                "No fine-tuned model/adapter found in %s.  "
                "Using base model: %s", OUTPUT_DIR, MODEL_NAME
            )

    # Resolve tokenizer path
    if os.path.exists(model_path):
        # Look for tokenizer configuration in model path
        if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            tokenizer_path = model_path
        elif os.path.exists(os.path.join(OUTPUT_DIR, "tokenizer", "tokenizer_config.json")):
            tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer")

    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_NAME or MODEL_NAME

    # 2. Load tokenizer
    logger.info("Loading tokenizer from: %s", tokenizer_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=TRUST_REMOTE_CODE,
            use_fast=USE_FAST_TOKENIZER,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        raise

    # Handle padding side for generation
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    # 3. Load model
    # Determine precision/dtype
    if MIXED_PRECISION == "bf16":
        torch_dtype = torch.bfloat16
    elif MIXED_PRECISION == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info("Loading model/adapter from: %s", model_path)
    log_gpu_memory("before inference model load")

    # Check if the target is a LoRA adapter directory
    is_lora_dir = False
    if os.path.exists(model_path) and os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            is_lora_dir = True

    try:
        load_kwargs = dict(
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR,
        )

        if ENABLE_QLORA and BNB_AVAILABLE:
            logger.info("Loading model with QLoRA 4-bit quantisation for inference.")
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            qlora_compute_dt = compute_dtype_map.get(QLORA_COMPUTE_DTYPE, torch.float16)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=QLORA_QUANT_TYPE,
                bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
                bnb_4bit_compute_dtype=qlora_compute_dt,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        if is_lora_dir:
            # We must load the base model first, then apply PEFT
            logger.info("Loading base model: %s", MODEL_NAME)
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
            if not (ENABLE_QLORA and BNB_AVAILABLE):
                base_model = base_model.to(device)

            logger.info("Loading LoRA adapter: %s", model_path)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load as standard CausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            if not (ENABLE_QLORA and BNB_AVAILABLE):
                model = model.to(device)

        model.eval()  # Set to evaluation mode

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise

    log_gpu_memory("after inference model load")
    logger.info("Model loaded and ready for generation.")

    # 4. Generate response function
    def _generate_response(user_prompt):
        # Format prompt according to dataset formats if applicable.
        if "### Instruction:" not in user_prompt and not user_prompt.startswith("<|im_start|>") and not user_prompt.startswith("[INST]"):
            formatted_prompt = "### Instruction:\n{}\n\n### Response:\n".format(user_prompt)
        else:
            formatted_prompt = user_prompt

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
            do_sample=INFERENCE_DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if INFERENCE_DO_SAMPLE:
            generation_kwargs["temperature"] = INFERENCE_TEMPERATURE
            generation_kwargs["top_p"] = INFERENCE_TOP_P
            generation_kwargs["repetition_penalty"] = INFERENCE_REPETITION_PENALTY

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        # Decode output
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response

    # 5. Execute prompt or run interactive chat loop
    if INFERENCE_PROMPT.lower() == "interactive":
        logger.info("Entering interactive chat loop. Type 'exit' or 'quit' to end.")
        print("-" * 72)
        print("Model is ready! Ask your questions:")
        print("-" * 72)

        while True:
            try:
                user_input = input("\nUser > ")
                if user_input.strip().lower() in ("exit", "quit"):
                    print("Exiting chat loop...")
                    break
                if not user_input.strip():
                    continue

                print("Model > ", end="", flush=True)
                response = _generate_response(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\nExiting chat loop...")
                break
            except Exception as e:
                logger.error("Error generating response: %s", e)
    else:
        logger.info("Generating response for prompt: %s", INFERENCE_PROMPT)
        try:
            response = _generate_response(INFERENCE_PROMPT)
            print("\n" + "=" * 72)
            print("Prompt   : {}".format(INFERENCE_PROMPT))
            print("Response : {}".format(response))
            print("=" * 72 + "\n")
        except Exception as e:
            logger.error("Error generating response: %s", e)

    # Cleanup
    cleanup_memory()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — MAIN ENTRYPOINT                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main():
    """
    Main orchestration function.

    Pipeline:
      1. Validate configuration.
      2. Set random seeds for reproducibility.
      3. Load and configure the tokenizer.
      4. Load the base model.
      5. Apply LoRA adapters.
      6. Load, format, and tokenize the dataset.
      7. Build the Trainer.
      8. Train with OOM recovery.
      9. Save results.
      10. Final summary.
    """
    start_time = time.time()

    try:
        # ── Step 1: Validate configuration ───────────────────────────────
        logger.info("Step 1/9: Validating configuration...")
        validate_config()

        # ── Step 2: Set seeds ────────────────────────────────────────────
        logger.info("Step 2/9: Setting random seed to %d...", SEED)
        set_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # ── Step 2B: Route to Inference if configured ────────────────────
        if RUN_MODE == "inference":
            run_inference()
            return

        # ── Step 3: Select device ────────────────────────────────────────
        device = get_device()

        # ── Step 4: Load tokenizer ───────────────────────────────────────
        logger.info("Step 3/9: Loading tokenizer...")
        tokenizer = setup_tokenizer(MODEL_NAME)

        # ── Step 5: Load model ───────────────────────────────────────────
        logger.info("Step 4/9: Loading model...")
        model, model_type = load_model(MODEL_NAME, device, tokenizer)

        # ── Step 6: Apply LoRA ───────────────────────────────────────────
        logger.info("Step 5/9: Configuring LoRA...")
        model, lora_config = setup_lora(model, model_type)

        # ── Step 7: Prepare dataset ──────────────────────────────────────
        logger.info("Step 6/9: Preparing dataset...")
        tokenized_dataset = prepare_dataset(tokenizer)

        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset.get("validation", None)

        logger.info("Training examples : %d", len(train_dataset))
        if eval_dataset is not None:
            logger.info("Validation examples: %d", len(eval_dataset))

        # ── Step 8: Determine batch size ─────────────────────────────────
        if AUTO_BATCH_SIZE:
            batch_size = estimate_safe_batch_size(
                model, MAX_SEQ_LENGTH, MIXED_PRECISION,
            )
            logger.info("Auto-estimated batch size: %d", batch_size)
        else:
            batch_size = BATCH_SIZE
            logger.info("Using configured batch size: %d", batch_size)

        # ── Step 9: Build Trainer ────────────────────────────────────────
        logger.info("Step 7/9: Building Trainer...")
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=batch_size,
        )

        # ── Step 10: Train ───────────────────────────────────────────────
        logger.info("Step 8/9: Starting training...")
        cleanup_memory()
        train_with_oom_recovery(trainer)

        # ── Step 11: Save ────────────────────────────────────────────────
        logger.info("Step 9/9: Saving results...")
        save_results(model, tokenizer)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(130)

    except Exception as e:
        logger.error("Fatal error: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        # ── Final summary ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("=" * 72)
        logger.info("  TRAINING COMPLETE")
        logger.info(
            "  Total time : %dh %dm %.1fs", int(hours), int(minutes), seconds,
        )
        logger.info("  Output dir : %s", os.path.abspath(OUTPUT_DIR))
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info("  Peak VRAM  : %.2f GB", peak_mem)
        logger.info("=" * 72)
        cleanup_memory()


# ── Script entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

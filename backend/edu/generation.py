"""
backend/edu/generation.py
────────────────────────────────────────────────────────────
Answer generation service using Qwen2.5-3B-Instruct.

Responsible for:
    1. Loading Qwen in 4-bit quantization (BitsAndBytes)
    2. Providing a generate_fn(prompt) -> str callable
    3. Managing model lifecycle

VRAM usage:
    - Qwen 3B 4-bit: ~2.5 GB
    - KV cache during generation: ~1-2 GB
    - Total: ~4 GB on RTX 3090 (24 GB)

Public API:
    load_qwen()       -> (model, tokenizer)
    create_generate_fn() -> Callable[[str], str]
    unload_qwen()     -> None
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level references for lifecycle management
_model = None
_tokenizer = None


def load_qwen(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    use_4bit: bool = True,
) -> tuple:
    """
    Load Qwen2.5 model and tokenizer.

    Args:
        model_name: HuggingFace model ID
        device:     "auto", "cuda", or "cpu"
        use_4bit:   Whether to use 4-bit quantization (recommended)

    Returns:
        (model, tokenizer) tuple
    """
    global _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading Qwen: %s (4bit=%s)", model_name, use_4bit)

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Quantization config
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            logger.warning(
                "bitsandbytes not available -- loading in float16 without quantization. "
                "Install with: pip install bitsandbytes"
            )
            model_kwargs["device_map"] = device if device != "auto" else "auto"
    else:
        model_kwargs["device_map"] = device if device != "auto" else "auto"

    _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    _model.eval()

    # Log memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Qwen loaded. GPU memory used: %.2f GB", mem_gb)
    else:
        logger.info("Qwen loaded on CPU")

    return _model, _tokenizer


def create_generate_fn(
    model=None,
    tokenizer=None,
    max_new_tokens: int = 350,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> Callable[[str], str]:
    """
    Create a generate function that wraps Qwen inference.

    The returned function has signature: fn(prompt: str) -> str

    Args:
        model:          Loaded Qwen model (uses module-level if None)
        tokenizer:      Loaded tokenizer (uses module-level if None)
        max_new_tokens: Maximum tokens to generate
        temperature:    Sampling temperature (low = deterministic)
        do_sample:      Whether to sample (False = greedy)

    Returns:
        Callable that takes a prompt string and returns generated text
    """
    mdl = model or _model
    tok = tokenizer or _tokenizer

    if mdl is None or tok is None:
        raise RuntimeError("Qwen not loaded. Call load_qwen() first.")

    def generate(prompt: str) -> str:
        """Generate a response from Qwen given a prompt."""
        # Format as chat message for instruct model
        messages = [
            {"role": "system", "content": "You are a helpful educational AI assistant."},
            {"role": "user", "content": prompt},
        ]

        # Use the chat template
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tok(text, return_tensors="pt").to(mdl.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tok.eos_token_id,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = outputs[0][input_len:]
        response = tok.decode(generated_ids, skip_special_tokens=True).strip()

        logger.debug("Generated %d tokens", len(generated_ids))
        return response

    return generate


def unload_qwen() -> None:
    """Explicitly unload Qwen model and free VRAM."""
    global _model, _tokenizer

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Qwen unloaded, VRAM freed")
    else:
        logger.info("Qwen unloaded")


def is_loaded() -> bool:
    """Check if Qwen is currently loaded."""
    return _model is not None and _tokenizer is not None

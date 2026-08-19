"""
backend/edu/generation.py
────────────────────────────────────────────────────────────
Answer generation service using Qwen2.5-VL-7B-Instruct.

Responsible for:
    1. Loading Qwen VL in 4-bit quantization (BitsAndBytes)
    2. Providing a generate_fn(prompt, image_paths) -> str callable
    3. Managing model lifecycle

VRAM usage:
    - Qwen2.5-VL-7B 4-bit: ~5-6 GB
    - KV cache during generation: ~1-2 GB
    - Total: ~7-8 GB on RTX 3090 (24 GB)

Public API:
    load_qwen()       -> (model, processor)
    create_generate_fn() -> Callable
    unload_qwen()     -> None
"""
from __future__ import annotations

import os
from typing import Callable, Optional
from pathlib import Path

import torch
from PIL import Image

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level references for lifecycle management
_model = None
_processor = None


def load_qwen(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "auto",
    use_4bit: bool = True,
) -> tuple:
    """
    Load Qwen2.5-VL model and processor.

    Args:
        model_name: HuggingFace model ID
        device:     "auto", "cuda", or "cpu"
        use_4bit:   Whether to use 4-bit quantization (recommended)

    Returns:
        (model, processor) tuple
    """
    global _model, _processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    logger.info("Loading Qwen VL: %s (4bit=%s)", model_name, use_4bit)

    # Load processor
    _processor = AutoProcessor.from_pretrained(
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

    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    _model.eval()

    # Log memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Qwen VL loaded. GPU memory used: %.2f GB", mem_gb)
    else:
        logger.info("Qwen VL loaded on CPU")

    return _model, _processor


def create_generate_fn(
    model=None,
    processor=None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> Callable:
    """
    Create a generate function that wraps Qwen VL inference.

    The returned function has signature: fn(prompt: str, image_paths: list[str] | None = None) -> str

    Args:
        model:          Loaded Qwen model (uses module-level if None)
        processor:      Loaded processor (uses module-level if None)
        max_new_tokens: Maximum tokens to generate
        temperature:    Sampling temperature (low = deterministic)
        do_sample:      Whether to sample (False = greedy)

    Returns:
        Callable that takes a prompt string (and optional image paths) and returns generated text
    """
    mdl = model or _model
    proc = processor or _processor

    if mdl is None or proc is None:
        raise RuntimeError("Qwen not loaded. Call load_qwen() first.")

    from qwen_vl_utils import process_vision_info

    def generate(prompt: str, image_paths: list[str] | None = None) -> str:
        """Generate a response from Qwen VL given a prompt and optional images."""
        
        # Build the user content block
        user_content = []
        if image_paths:
            for path in image_paths:
                if os.path.exists(path):
                    # Using local file path format expected by qwen_vl_utils
                    user_content.append({"type": "image", "image": f"file://{path}"})
        
        user_content.append({"type": "text", "text": prompt})

        # Format as chat message for instruct model
        messages = [
            {"role": "system", "content": "You are a helpful educational AI assistant."},
            {"role": "user", "content": user_content},
        ]

        # Use the chat template
        text = proc.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(mdl.device)

        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=0.9 if do_sample else 1.0,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = proc.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        logger.debug("Generated response of length: %d chars", len(response))
        return response

    return generate


def unload_qwen() -> None:
    """Explicitly unload Qwen model and free VRAM."""
    global _model, _processor

    if _model is not None:
        del _model
        _model = None

    if _processor is not None:
        del _processor
        _processor = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Qwen VL unloaded, VRAM freed")
    else:
        logger.info("Qwen VL unloaded")


def is_loaded() -> bool:
    """Check if Qwen is currently loaded."""
    return _model is not None and _processor is not None

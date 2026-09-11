from __future__ import annotations
import gc
import os
from typing import Optional, Tuple

import psutil

from config import get_config
from core.exceptions import InsufficientMemoryError, ModelLoadError
from core.hardware import detect_hardware, EXECUTION_MODE_DEMO
from core.logger import get_logger

logger = get_logger(__name__)

_loaded_model = None
_loaded_tokenizer = None
_loaded_model_id = None


def _available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _require_ram(min_gb: float) -> None:
    available = _available_ram_gb()
    if available < min_gb:
        raise InsufficientMemoryError(
            f"Insufficient RAM: {available:.2f} GB available, {min_gb:.2f} GB required"
        )


def load_model_and_tokenizer(
    model_id: Optional[str] = None,
    force_cpu: bool = True,
    min_ram_gb: float = 1.5,
) -> Tuple[object, object]:
    global _loaded_model, _loaded_tokenizer, _loaded_model_id

    cfg = get_config()
    if model_id is None:
        model_id = cfg.local_model_id

    if _loaded_model is not None and _loaded_model_id == model_id:
        return _loaded_model, _loaded_tokenizer

    _require_ram(min_ram_gb)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ModelLoadError("transformers library not available") from exc

    try:
        logger.info(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=cfg.hf_token or None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=cfg.hf_token or None,
            low_cpu_mem_usage=True,
        )
        model.eval()

        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _loaded_model_id = model_id

        logger.info(f"Model loaded successfully: {model_id}")
        return model, tokenizer

    except InsufficientMemoryError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model '{model_id}': {exc}") from exc


def unload_model() -> None:
    global _loaded_model, _loaded_tokenizer, _loaded_model_id
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_model_id = None
    gc.collect()
    logger.info("Model unloaded and memory released")


def is_model_loaded() -> bool:
    return _loaded_model is not None


def get_loaded_model_id() -> Optional[str]:
    return _loaded_model_id


def get_model_info() -> dict:
    if _loaded_model is None:
        return {"loaded": False, "model_id": None, "parameters": None}

    try:
        total_params = sum(p.numel() for p in _loaded_model.parameters())
        trainable_params = sum(p.numel() for p in _loaded_model.parameters() if p.requires_grad)
    except Exception:
        total_params = None
        trainable_params = None

    return {
        "loaded": True,
        "model_id": _loaded_model_id,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }

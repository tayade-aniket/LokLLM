from __future__ import annotations
from typing import Any, Optional

from config import get_config
from core.exceptions import ModelLoadError
from core.logger import get_logger

logger = get_logger(__name__)


def build_lora_config(
    r: Optional[int] = None,
    alpha: Optional[int] = None,
    dropout: Optional[float] = None,
    target_modules: Optional[list] = None,
    task_type: str = "CAUSAL_LM",
) -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ModelLoadError("peft library not available") from exc

    cfg = get_config()
    r = r if r is not None else cfg.lora_r
    alpha = alpha if alpha is not None else cfg.lora_alpha
    dropout = dropout if dropout is not None else cfg.lora_dropout
    target_modules = target_modules or cfg.lora_target_modules

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model: Any, lora_config: Any) -> Any:
    try:
        from peft import get_peft_model
    except ImportError as exc:
        raise ModelLoadError("peft library not available") from exc

    peft_model = get_peft_model(model, lora_config)
    trainable, total = count_lora_parameters(peft_model)
    logger.info(f"LoRA applied. Trainable: {trainable:,} / {total:,} parameters")
    return peft_model


def count_lora_parameters(model: Any) -> tuple[int, int]:
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable, total
    except Exception:
        return 0, 0


def build_bnb_config() -> Any:
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ModelLoadError("bitsandbytes or transformers not available") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def get_lora_info(cfg=None) -> dict:
    if cfg is None:
        cfg = get_config()
    return {
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "target_modules": cfg.lora_target_modules,
        "quantization": "4-bit NF4 (cloud only)",
        "double_quantization": True,
    }

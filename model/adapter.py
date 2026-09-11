from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional

from core.exceptions import AdapterError
from core.logger import get_logger

logger = get_logger(__name__)


def get_adapter_path(adapters_dir: str, adapter_name: str) -> str:
    return os.path.join(adapters_dir, adapter_name)


def save_adapter_metadata(adapter_dir: str, metadata: dict) -> str:
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    meta_path = os.path.join(adapter_dir, "adapter_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


def load_adapter_metadata(adapter_dir: str) -> dict:
    meta_path = os.path.join(adapter_dir, "adapter_metadata.json")
    if not Path(meta_path).exists():
        raise AdapterError(f"Adapter metadata not found at: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_peft_adapter(model: Any, adapter_dir: str, metadata: Optional[dict] = None) -> str:
    try:
        Path(adapter_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        if metadata:
            save_adapter_metadata(adapter_dir, metadata)
        logger.info(f"Adapter saved to: {adapter_dir}")
        return adapter_dir
    except Exception as exc:
        raise AdapterError(f"Failed to save adapter: {exc}") from exc


def load_peft_adapter(model: Any, adapter_dir: str) -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise AdapterError("peft library not available") from exc

    if not Path(adapter_dir).exists():
        raise AdapterError(f"Adapter directory not found: {adapter_dir}")

    try:
        peft_model = PeftModel.from_pretrained(model, adapter_dir)
        logger.info(f"Adapter loaded from: {adapter_dir}")
        return peft_model
    except Exception as exc:
        raise AdapterError(f"Failed to load adapter from '{adapter_dir}': {exc}") from exc


def get_adapter_info(adapter_dir: str) -> dict:
    if not Path(adapter_dir).exists():
        return {"exists": False, "path": adapter_dir}

    files = list(Path(adapter_dir).rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    info = {
        "exists": True,
        "path": adapter_dir,
        "file_count": len([f for f in files if f.is_file()]),
        "total_size_bytes": total_size,
        "total_size_kb": round(total_size / 1024, 2),
    }

    try:
        metadata = load_adapter_metadata(adapter_dir)
        info["metadata"] = metadata
    except AdapterError:
        info["metadata"] = None

    return info


def list_adapters(adapters_dir: str) -> list[dict]:
    base = Path(adapters_dir)
    if not base.exists():
        return []

    result = []
    for item in base.iterdir():
        if item.is_dir():
            result.append(get_adapter_info(str(item)))
    return result


def create_demo_adapter_record(adapters_dir: str, round_num: int = 1) -> str:
    adapter_name = f"demo_fedavg_round_{round_num}"
    adapter_dir = os.path.join(adapters_dir, adapter_name)
    metadata = {
        "adapter_name": adapter_name,
        "round": round_num,
        "mode": "DEMO",
        "note": "This is a simulated adapter record. No actual training was performed on this device.",
        "trainable_parameters": 49152,
        "total_parameters": 85000000,
        "adapter_size_kb": 192.0,
        "clients": ["client_1", "client_2", "client_3"],
        "strategy": "FedAvg",
        "dp_enabled": True,
    }
    save_adapter_metadata(adapter_dir, metadata)
    return adapter_dir

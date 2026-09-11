from __future__ import annotations
from typing import Any, Optional

from core.logger import get_logger

logger = get_logger(__name__)


def build_flower_server_config(num_rounds: int = 1) -> Any:
    try:
        from flwr.server import ServerConfig
        return ServerConfig(num_rounds=num_rounds)
    except ImportError as exc:
        raise ImportError("flwr not available. Server runs in cloud environment.") from exc


def get_server_info(num_rounds: int = 1) -> dict:
    return {
        "framework": "Flower (flwr)",
        "strategy": "FedAvg",
        "num_rounds": num_rounds,
        "num_clients": 3,
        "environment": "Cloud GPU (not local)",
    }

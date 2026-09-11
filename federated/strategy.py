from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

from core.logger import get_logger

logger = get_logger(__name__)


def build_fedavg_strategy(
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 3,
    min_evaluate_clients: int = 3,
    min_available_clients: int = 3,
) -> Any:
    try:
        from flwr.server.strategy import FedAvg
    except ImportError as exc:
        raise ImportError("flwr not available. Federated simulation requires cloud environment.") from exc

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
    )
    logger.info("FedAvg strategy created")
    return strategy


def get_strategy_info() -> dict:
    return {
        "name": "FedAvg",
        "description": "Federated Averaging — McMahan et al. 2017",
        "aggregation": "Weighted parameter averaging",
        "clients": 3,
        "fraction_fit": 1.0,
    }

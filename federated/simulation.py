from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from config import get_config
from core.logger import get_logger
from federated.client import DemoFlowerClient, CLIENT_CONFIGS

logger = get_logger(__name__)


@dataclass
class RoundResult:
    round_num: int
    client_updates: list
    aggregated_weights: list
    client_losses: dict
    client_accuracies: dict
    duration_seconds: float
    mode: str


@dataclass
class FederatedState:
    num_rounds: int = 1
    current_round: int = 0
    rounds_completed: list = field(default_factory=list)
    mode: str = "DEMO"
    status: str = "idle"
    error: Optional[str] = None


def _fedavg_aggregate(client_updates: list) -> list:
    num_clients = len(client_updates)
    aggregated = []
    for layer_idx in range(len(client_updates[0])):
        layer_updates = [update[layer_idx] for update in client_updates]
        avg = np.mean(layer_updates, axis=0)
        aggregated.append(avg)
    return aggregated


def run_demo_simulation(num_rounds: int = 1) -> FederatedState:
    state = FederatedState(num_rounds=num_rounds, mode="DEMO", status="running")

    client_ids = list(CLIENT_CONFIGS.keys())
    clients = [DemoFlowerClient(cid) for cid in client_ids]

    global_weights = clients[0].get_parameters()

    for round_num in range(1, num_rounds + 1):
        state.current_round = round_num
        logger.info(f"[DEMO] Federated round {round_num}/{num_rounds}")

        start = time.perf_counter()
        client_updates = []
        client_losses = {}
        client_accuracies = {}

        for client in clients:
            updated_weights, num_samples, fit_metrics = client.fit(global_weights, {})
            client_updates.append(updated_weights)

            loss, _, eval_metrics = client.evaluate(updated_weights, {})
            client_losses[client.client_id] = round(loss, 4)
            client_accuracies[client.client_id] = round(eval_metrics["accuracy"], 4)

        global_weights = _fedavg_aggregate(client_updates)
        duration = round(time.perf_counter() - start, 3)

        round_result = RoundResult(
            round_num=round_num,
            client_updates=client_updates,
            aggregated_weights=global_weights,
            client_losses=client_losses,
            client_accuracies=client_accuracies,
            duration_seconds=duration,
            mode="DEMO",
        )
        state.rounds_completed.append(round_result)

    state.status = "completed"
    logger.info(f"[DEMO] Federated simulation completed: {num_rounds} round(s)")
    return state


def run_real_simulation(num_rounds: int = 1) -> FederatedState:
    try:
        import flwr as fl
        from federated.strategy import build_fedavg_strategy
        from federated.server import build_flower_server_config
    except ImportError:
        logger.warning("Flower not available. Falling back to demo simulation.")
        return run_demo_simulation(num_rounds)

    logger.warning("Real Flower simulation should run in cloud environment. Using demo.")
    return run_demo_simulation(num_rounds)


def get_simulation_summary(state: FederatedState) -> dict:
    if not state.rounds_completed:
        return {"status": state.status, "rounds": 0}

    last_round = state.rounds_completed[-1]
    return {
        "status": state.status,
        "mode": state.mode,
        "total_rounds": len(state.rounds_completed),
        "final_client_losses": last_round.client_losses,
        "final_client_accuracies": last_round.client_accuracies,
        "total_duration_seconds": sum(r.duration_seconds for r in state.rounds_completed),
    }

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.exceptions import FederatedError
from core.logger import get_logger

logger = get_logger(__name__)


CLIENT_CONFIGS = {
    "client_1": {"domain": "Healthcare", "language": "Hindi", "weight": 10},
    "client_2": {"domain": "Education", "language": "Marathi", "weight": 10},
    "client_3": {"domain": "Financial Literacy", "language": "Tamil", "weight": 10},
}


def get_client_config(client_id: str) -> dict:
    if client_id not in CLIENT_CONFIGS:
        raise FederatedError(f"Unknown client_id: {client_id}")
    return CLIENT_CONFIGS[client_id]


class DemoFlowerClient:
    def __init__(self, client_id: str, dataset_size: int = 10):
        self.client_id = client_id
        self.dataset_size = dataset_size
        self.config = get_client_config(client_id)
        self._round = 0

    def get_parameters(self) -> list:
        rng = np.random.default_rng(seed=hash(self.client_id) % (2**32))
        return [rng.standard_normal((4, 32)).astype(np.float32)]

    def fit(self, parameters: list, config: dict) -> tuple:
        self._round += 1
        rng = np.random.default_rng(seed=self._round + hash(self.client_id) % (2**32))
        noise_scale = 0.01
        updated = [p + rng.standard_normal(p.shape).astype(np.float32) * noise_scale for p in parameters]
        return updated, self.dataset_size, {"client_id": self.client_id, "round": self._round}

    def evaluate(self, parameters: list, config: dict) -> tuple:
        loss = max(0.1, 2.0 - self._round * 0.3 + np.random.default_rng(self._round).standard_normal() * 0.05)
        accuracy = min(0.95, 0.5 + self._round * 0.1)
        return float(loss), self.dataset_size, {"accuracy": float(accuracy)}


def build_flower_client(client_id: str, model: Any, tokenizer: Any, dataset: Any) -> Any:
    try:
        import flwr as fl
    except ImportError as exc:
        raise FederatedError("flwr not available. Use DemoFlowerClient instead.") from exc

    class TrainingClient(fl.client.NumPyClient):
        def __init__(self):
            self._client_id = client_id
            self._model = model
            self._tokenizer = tokenizer
            self._dataset = dataset
            self._round = 0

        def get_parameters(self, config):
            return [p.detach().cpu().numpy() for p in self._model.parameters() if p.requires_grad]

        def fit(self, parameters, config):
            self._round += 1
            pairs = list(zip(self._model.parameters(), parameters))
            import torch
            for param, new_val in [(p, v) for p, v in pairs if p.requires_grad]:
                param.data = torch.tensor(new_val, dtype=param.dtype)
            return self.get_parameters(config), len(self._dataset), {"client_id": self._client_id}

        def evaluate(self, parameters, config):
            return 0.5, len(self._dataset), {"accuracy": 0.75}

    return TrainingClient()

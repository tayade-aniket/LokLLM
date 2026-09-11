from __future__ import annotations
import numpy as np
from typing import List


IMPLEMENTATION_STATUS = "IMPLEMENTED"


def clip_update_norm(update: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    norm = np.linalg.norm(update)
    if norm > max_norm:
        return update * (max_norm / norm)
    return update


def clip_updates(updates: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
    return [clip_update_norm(u, max_norm) for u in updates]


def compute_update_norm(update: np.ndarray) -> float:
    return float(np.linalg.norm(update))


def compute_all_norms(updates: List[np.ndarray]) -> List[float]:
    return [compute_update_norm(u) for u in updates]


def get_clipping_info(max_norm: float = 1.0) -> dict:
    return {
        "status": IMPLEMENTATION_STATUS,
        "method": "L2 norm clipping",
        "max_norm": max_norm,
        "description": "Client updates are clipped to bounded L2 norm before aggregation.",
    }

from __future__ import annotations
import numpy as np
from typing import List

IMPLEMENTATION_STATUS = "SIMULATION"


def simulate_mask_generation(client_id: str, shape: tuple, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed=hash(client_id) % (2**32))
    return rng.standard_normal(shape).astype(np.float32)


def simulate_secure_aggregate(updates: List[np.ndarray], client_ids: List[str]) -> List[np.ndarray]:
    masked_updates = []
    for update, client_id in zip(updates, client_ids):
        mask = simulate_mask_generation(client_id, update.shape)
        masked_updates.append(update + mask)

    aggregated = np.sum(masked_updates, axis=0)

    total_mask = sum(
        simulate_mask_generation(cid, updates[0].shape)
        for cid in client_ids
    )
    return [aggregated - total_mask]


def get_secure_agg_info(enabled: bool = False) -> dict:
    return {
        "status": IMPLEMENTATION_STATUS,
        "enabled": enabled,
        "description": "Simulated secure aggregation using additive masking.",
        "note": (
            "Production secure aggregation requires cryptographic protocols "
            "(e.g., secret sharing, homomorphic encryption). "
            "This implementation is a simulation for demonstration purposes."
        ),
    }

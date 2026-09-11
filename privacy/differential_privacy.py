from __future__ import annotations
import numpy as np
from typing import List, Optional

from core.exceptions import PrivacyConfigError

IMPLEMENTATION_STATUS = "DEMONSTRATION"


def validate_dp_params(epsilon: float, delta: float, noise_multiplier: float) -> None:
    if epsilon <= 0:
        raise PrivacyConfigError(f"epsilon must be positive, got {epsilon}")
    if not (0 < delta < 1):
        raise PrivacyConfigError(f"delta must be in (0, 1), got {delta}")
    if noise_multiplier <= 0:
        raise PrivacyConfigError(f"noise_multiplier must be positive, got {noise_multiplier}")


def add_gaussian_noise(
    update: np.ndarray,
    noise_multiplier: float,
    max_norm: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = noise_multiplier * max_norm
    noise = rng.normal(0, sigma, update.shape).astype(update.dtype)
    return update + noise


def apply_dp_to_updates(
    updates: List[np.ndarray],
    noise_multiplier: float,
    max_norm: float,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    return [add_gaussian_noise(u, noise_multiplier, max_norm, seed) for u in updates]


def estimate_privacy_cost(
    noise_multiplier: float,
    num_rounds: int,
    dataset_size: int,
    batch_size: int,
    delta: float,
) -> dict:
    try:
        q = batch_size / dataset_size
        epsilon_approx = q * noise_multiplier * (2 * num_rounds) ** 0.5
        return {
            "epsilon_approx": round(epsilon_approx, 4),
            "delta": delta,
            "note": "Approximate estimate. Not a formal privacy guarantee.",
            "status": IMPLEMENTATION_STATUS,
        }
    except Exception:
        return {
            "epsilon_approx": None,
            "delta": delta,
            "note": "Could not estimate privacy cost.",
            "status": IMPLEMENTATION_STATUS,
        }


def get_dp_info(cfg=None) -> dict:
    if cfg is None:
        from config import get_config
        cfg = get_config()
    return {
        "status": IMPLEMENTATION_STATUS,
        "enabled": cfg.dp_enabled,
        "mechanism": "Gaussian noise addition (demonstration)",
        "epsilon": cfg.dp_epsilon,
        "delta": cfg.dp_delta,
        "noise_multiplier": cfg.dp_noise_multiplier,
        "max_grad_norm": cfg.dp_max_grad_norm,
        "note": "This is a demonstration of the DP mechanism. Production-grade DP requires formal mathematical verification.",
    }

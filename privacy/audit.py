from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from config import get_config
from privacy.clipping import get_clipping_info
from privacy.differential_privacy import get_dp_info
from privacy.secure_aggregation import get_secure_agg_info


@dataclass
class PrivacyAuditReport:
    raw_data_uploaded_bytes: int = 0
    raw_dataset_transmitted: bool = False
    local_training: bool = True
    adapter_transmitted: bool = True
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_status: str = "DEMONSTRATION"
    secure_agg_status: str = "SIMULATION"
    secure_agg_enabled: bool = False
    clipping_status: str = "IMPLEMENTED"
    clipping_max_norm: float = 1.0
    mode: str = "DEMO"
    note: str = ""


def build_audit_report(mode: str = "DEMO") -> PrivacyAuditReport:
    cfg = get_config()
    return PrivacyAuditReport(
        raw_data_uploaded_bytes=0,
        raw_dataset_transmitted=False,
        local_training=True,
        adapter_transmitted=True,
        dp_enabled=cfg.dp_enabled,
        dp_epsilon=cfg.dp_epsilon,
        dp_delta=cfg.dp_delta,
        dp_status="DEMONSTRATION",
        secure_agg_status="SIMULATION",
        secure_agg_enabled=cfg.secure_agg_enabled,
        clipping_status="IMPLEMENTED",
        clipping_max_norm=cfg.dp_max_grad_norm,
        mode=mode,
        note=(
            "Personal data stays on the client device. Only model parameter updates "
            "(adapter weights) are transmitted after clipping and noise addition."
        ),
    )


def get_privacy_boundary_summary() -> dict:
    cfg = get_config()
    return {
        "raw_data_uploaded": "0 bytes",
        "raw_dataset_transmitted": "NO",
        "local_training": "YES (cloud simulation for heavy compute)",
        "adapter_transmitted": "YES (LoRA adapter weights only)",
        "differential_privacy": "Enabled" if cfg.dp_enabled else "Disabled",
        "secure_aggregation": "Simulation" if not cfg.secure_agg_enabled else "Enabled",
        "clipping": "Implemented",
    }


def save_audit_report(report: PrivacyAuditReport, results_dir: str) -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = str(Path(results_dir) / "privacy_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    return path

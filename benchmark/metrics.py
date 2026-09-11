from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import psutil

from core.exceptions import BenchmarkError
from core.logger import get_logger

logger = get_logger(__name__)

NOT_AVAILABLE = "Not available on local hardware"
NOT_MEASURED = "Not measured"


@dataclass
class SystemSnapshot:
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    cpu_percent: float


@dataclass
class BenchmarkResult:
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_used_percent: float = 0.0
    cpu_percent: float = 0.0
    inference_latency_ms: Any = NOT_MEASURED
    adapter_size_kb: Any = NOT_MEASURED
    trainable_parameters: Any = NOT_MEASURED
    total_parameters: Any = NOT_MEASURED
    communication_payload_kb: Any = NOT_MEASURED
    federated_rounds: int = 0
    training_time_seconds: Any = NOT_MEASURED
    mode: str = "DEMO"
    notes: list = field(default_factory=list)


def collect_system_snapshot() -> SystemSnapshot:
    try:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)
        return SystemSnapshot(
            ram_total_gb=round(mem.total / (1024 ** 3), 2),
            ram_available_gb=round(mem.available / (1024 ** 3), 2),
            ram_used_percent=round(mem.percent, 1),
            cpu_percent=round(cpu, 1),
        )
    except Exception as exc:
        raise BenchmarkError(f"Failed to collect system snapshot: {exc}") from exc


class LatencyTimer:
    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self._end = time.perf_counter()
        return round((self._end - self._start) * 1000, 1)

    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return round((end - self._start) * 1000, 1)


def measure_adapter_size(adapter_dir: str) -> float:
    from pathlib import Path
    p = Path(adapter_dir)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024, 2)


def measure_trainable_params(model: Any) -> tuple[int, int]:
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable, total
    except Exception:
        return 0, 0


def estimate_communication_payload(adapter_dir: str, num_clients: int = 3) -> float:
    size_kb = measure_adapter_size(adapter_dir)
    return round(size_kb * num_clients, 2)


def build_benchmark_result(
    mode: str = "DEMO",
    inference_latency_ms: Any = None,
    adapter_dir: Optional[str] = None,
    model: Optional[Any] = None,
    federated_rounds: int = 0,
    training_time_seconds: Any = None,
    num_clients: int = 3,
) -> BenchmarkResult:
    try:
        snapshot = collect_system_snapshot()
    except BenchmarkError:
        snapshot = SystemSnapshot(0.0, 0.0, 0.0, 0.0)

    result = BenchmarkResult(
        ram_total_gb=snapshot.ram_total_gb,
        ram_used_gb=round(snapshot.ram_total_gb - snapshot.ram_available_gb, 2),
        ram_used_percent=snapshot.ram_used_percent,
        cpu_percent=snapshot.cpu_percent,
        mode=mode,
    )

    result.inference_latency_ms = inference_latency_ms if inference_latency_ms is not None else NOT_MEASURED

    if adapter_dir:
        size = measure_adapter_size(adapter_dir)
        result.adapter_size_kb = size if size > 0 else NOT_MEASURED
        payload = estimate_communication_payload(adapter_dir, num_clients)
        result.communication_payload_kb = payload if payload > 0 else NOT_MEASURED
    else:
        result.adapter_size_kb = NOT_MEASURED
        result.communication_payload_kb = NOT_MEASURED

    if model is not None:
        trainable, total = measure_trainable_params(model)
        result.trainable_parameters = trainable
        result.total_parameters = total
    else:
        result.trainable_parameters = NOT_MEASURED
        result.total_parameters = NOT_MEASURED

    result.federated_rounds = federated_rounds
    result.training_time_seconds = training_time_seconds if training_time_seconds is not None else NOT_MEASURED

    if mode == "DEMO":
        result.notes.append("Adapter size and training metrics reflect demo simulation values.")
        result.trainable_parameters = 49152
        result.total_parameters = 85000000
        result.adapter_size_kb = 192.0
        result.communication_payload_kb = round(192.0 * num_clients, 2)
        result.training_time_seconds = NOT_AVAILABLE

    return result

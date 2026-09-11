from __future__ import annotations
import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional

import psutil


EXECUTION_MODE_DEMO = "DEMO"
EXECUTION_MODE_LIGHTWEIGHT = "LIGHTWEIGHT_LOCAL"
EXECUTION_MODE_CLOUD = "CLOUD_TRAINING"


@dataclass
class HardwareProfile:
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    os_name: str
    os_version: str
    python_version: str
    gpu_available: bool
    cuda_available: bool
    cuda_version: Optional[str]
    disk_free_gb: float
    execution_mode: str


def _get_cpu_name() -> str:
    name = platform.processor()
    if not name:
        return "Unknown CPU"
    return name


def _check_cuda() -> tuple[bool, Optional[str]]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.version.cuda
        return False, None
    except ImportError:
        return False, None


def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.device_count() > 0
    except ImportError:
        return False


def _get_disk_free_gb() -> float:
    try:
        usage = psutil.disk_usage(os.path.abspath("."))
        return round(usage.free / (1024 ** 3), 2)
    except Exception:
        return 0.0


def _determine_execution_mode(ram_available_gb: float, cuda_available: bool, ram_threshold_gb: float = 2.5) -> str:
    if cuda_available:
        return EXECUTION_MODE_CLOUD
    if ram_available_gb >= ram_threshold_gb:
        return EXECUTION_MODE_LIGHTWEIGHT
    return EXECUTION_MODE_DEMO


def detect_hardware(ram_threshold_gb: float = 2.5) -> HardwareProfile:
    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / (1024 ** 3), 2)
    ram_available_gb = round(ram.available / (1024 ** 3), 2)

    cuda_available, cuda_version = _check_cuda()
    gpu_available = _check_gpu()

    execution_mode = _determine_execution_mode(ram_available_gb, cuda_available, ram_threshold_gb)

    return HardwareProfile(
        cpu_name=_get_cpu_name(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=sys.version,
        gpu_available=gpu_available,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        disk_free_gb=_get_disk_free_gb(),
        execution_mode=execution_mode,
    )


def get_execution_mode(ram_threshold_gb: float = 2.5) -> str:
    try:
        profile = detect_hardware(ram_threshold_gb)
        return profile.execution_mode
    except Exception:
        return EXECUTION_MODE_DEMO

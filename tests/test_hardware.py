import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.hardware import detect_hardware, get_execution_mode, EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD, HardwareProfile


def test_detect_hardware_returns_profile():
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)


def test_hardware_profile_has_required_fields():
    profile = detect_hardware()
    assert isinstance(profile.cpu_name, str)
    assert len(profile.cpu_name) > 0
    assert profile.cpu_cores_logical >= 1
    assert profile.ram_total_gb > 0
    assert profile.ram_available_gb >= 0
    assert profile.ram_available_gb <= profile.ram_total_gb
    assert isinstance(profile.os_name, str)
    assert isinstance(profile.python_version, str)
    assert isinstance(profile.gpu_available, bool)
    assert isinstance(profile.cuda_available, bool)
    assert profile.disk_free_gb >= 0


def test_execution_mode_is_valid():
    profile = detect_hardware()
    assert profile.execution_mode in (EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD)


def test_get_execution_mode_returns_string():
    mode = get_execution_mode()
    assert isinstance(mode, str)
    assert mode in (EXECUTION_MODE_DEMO, EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD)


def test_low_ram_threshold_forces_demo():
    profile = detect_hardware(ram_threshold_gb=9999.0)
    assert profile.execution_mode in (EXECUTION_MODE_DEMO, EXECUTION_MODE_CLOUD)


def test_zero_ram_threshold_allows_lightweight():
    profile = detect_hardware(ram_threshold_gb=0.0)
    assert profile.execution_mode in (EXECUTION_MODE_LIGHTWEIGHT, EXECUTION_MODE_CLOUD)


def test_hardware_profile_cpu_cores_positive():
    profile = detect_hardware()
    assert profile.cpu_cores_physical >= 1
    assert profile.cpu_cores_logical >= profile.cpu_cores_physical


def test_ram_values_are_reasonable():
    profile = detect_hardware()
    assert 0.1 < profile.ram_total_gb < 2048
    assert profile.ram_available_gb >= 0

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


def test_config_loads():
    from config import get_config, AppConfig
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.app_name == "PRIVFEDQLORA"
    assert cfg.lora_r > 0
    assert cfg.lora_alpha > 0
    assert 0 < cfg.lora_dropout < 1
    assert cfg.local_batch_size >= 1
    assert cfg.max_seq_length > 0
    assert cfg.num_clients == 3


def test_hardware_detection_does_not_crash():
    from core.hardware import detect_hardware
    profile = detect_hardware()
    assert profile is not None


def test_exceptions_importable():
    from core.exceptions import (
        InsufficientMemoryError,
        ModelLoadError,
        AdapterError,
        DataGenerationError,
        FederatedError,
        PrivacyConfigError,
        BenchmarkError,
    )
    assert issubclass(InsufficientMemoryError, RuntimeError)
    assert issubclass(ModelLoadError, RuntimeError)


def test_synthetic_data_generates():
    from data.synthetic_data import get_all_clients_data
    data = get_all_clients_data()
    assert len(data) == 3
    for client_data in data:
        assert "samples" in client_data
        assert len(client_data["samples"]) >= 10


def test_model_loader_importable():
    from model.loader import is_model_loaded, get_model_info
    assert is_model_loaded() is False
    info = get_model_info()
    assert info["loaded"] is False


def test_inference_demo_mode_works():
    from model.inference import generate_demo_response, generate_response
    result = generate_demo_response("What is a fever?", domain="healthcare")
    assert "response" in result
    assert result["mode"] == "DEMO"
    assert "[DEMO]" in result["response"]

    result2 = generate_response("Tell me about savings", model=None, tokenizer=None)
    assert result2["mode"] == "DEMO"


def test_privacy_audit_builds():
    from privacy.audit import build_audit_report
    report = build_audit_report(mode="DEMO")
    assert report.raw_data_uploaded_bytes == 0
    assert report.raw_dataset_transmitted is False


def test_benchmark_metrics_collect():
    from benchmark.metrics import collect_system_snapshot
    snap = collect_system_snapshot()
    assert snap.ram_total_gb > 0
    assert snap.cpu_percent >= 0


def test_federated_demo_simulation_runs():
    from federated.simulation import run_demo_simulation
    state = run_demo_simulation(num_rounds=1)
    assert state.status == "completed"
    assert len(state.rounds_completed) == 1


def test_adapter_demo_record():
    import tempfile
    from model.adapter import create_demo_adapter_record, get_adapter_info
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = create_demo_adapter_record(tmpdir, round_num=1)
        info = get_adapter_info(adapter_dir)
        assert info["exists"] is True
        assert info["metadata"] is not None
        assert info["metadata"]["mode"] == "DEMO"


def test_lora_info_returns_config():
    from model.lora import get_lora_info
    info = get_lora_info()
    assert "r" in info
    assert "alpha" in info
    assert "dropout" in info


def test_logger_creates_logger():
    from core.logger import get_logger
    logger = get_logger("smoke_test")
    assert logger is not None
    logger.info("Smoke test logger OK")


def test_dataset_wrapper_loads():
    from data.synthetic_data import get_all_clients_data
    from data.dataset import LocalDataset
    all_data = get_all_clients_data()
    ds = LocalDataset.from_dict(all_data[0])
    assert len(ds) > 0
    summary = ds.summary()
    assert "client_id" in summary


def test_no_model_loaded_at_startup():
    from model.loader import is_model_loaded, get_loaded_model_id
    assert is_model_loaded() is False
    assert get_loaded_model_id() is None

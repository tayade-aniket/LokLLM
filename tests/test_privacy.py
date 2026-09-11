import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from privacy.clipping import clip_update_norm, clip_updates, compute_update_norm, compute_all_norms, get_clipping_info
from privacy.differential_privacy import (
    validate_dp_params,
    add_gaussian_noise,
    apply_dp_to_updates,
    estimate_privacy_cost,
    get_dp_info,
)
from privacy.secure_aggregation import simulate_mask_generation, simulate_secure_aggregate, get_secure_agg_info
from privacy.audit import build_audit_report, get_privacy_boundary_summary, save_audit_report


def test_clip_update_norm_within_bounds():
    update = np.array([3.0, 4.0])
    clipped = clip_update_norm(update, max_norm=1.0)
    assert compute_update_norm(clipped) <= 1.0 + 1e-6


def test_clip_update_norm_already_within():
    update = np.array([0.3, 0.4])
    clipped = clip_update_norm(update, max_norm=1.0)
    np.testing.assert_array_almost_equal(clipped, update)


def test_clip_updates_list():
    updates = [np.array([3.0, 4.0]), np.array([6.0, 8.0])]
    clipped = clip_updates(updates, max_norm=1.0)
    for c in clipped:
        assert compute_update_norm(c) <= 1.0 + 1e-6


def test_compute_update_norm():
    update = np.array([3.0, 4.0])
    norm = compute_update_norm(update)
    assert abs(norm - 5.0) < 1e-5


def test_compute_all_norms():
    updates = [np.array([3.0, 4.0]), np.array([0.0, 1.0])]
    norms = compute_all_norms(updates)
    assert len(norms) == 2
    assert abs(norms[0] - 5.0) < 1e-5


def test_get_clipping_info():
    info = get_clipping_info(max_norm=2.0)
    assert info["max_norm"] == 2.0
    assert info["status"] == "IMPLEMENTED"


def test_validate_dp_params_valid():
    validate_dp_params(1.0, 1e-5, 1.1)


def test_validate_dp_params_invalid_epsilon():
    from core.exceptions import PrivacyConfigError
    with pytest.raises(PrivacyConfigError):
        validate_dp_params(-1.0, 1e-5, 1.1)


def test_validate_dp_params_invalid_delta():
    from core.exceptions import PrivacyConfigError
    with pytest.raises(PrivacyConfigError):
        validate_dp_params(1.0, 1.5, 1.1)


def test_validate_dp_params_invalid_noise_multiplier():
    from core.exceptions import PrivacyConfigError
    with pytest.raises(PrivacyConfigError):
        validate_dp_params(1.0, 1e-5, -0.5)


def test_add_gaussian_noise_shape():
    update = np.zeros((4, 4), dtype=np.float32)
    noised = add_gaussian_noise(update, noise_multiplier=1.0, max_norm=1.0, seed=42)
    assert noised.shape == update.shape


def test_add_gaussian_noise_not_identical():
    update = np.zeros((4, 4), dtype=np.float32)
    noised = add_gaussian_noise(update, noise_multiplier=1.0, max_norm=1.0, seed=42)
    assert not np.array_equal(noised, update)


def test_apply_dp_to_updates():
    updates = [np.zeros((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)]
    noised = apply_dp_to_updates(updates, noise_multiplier=0.5, max_norm=1.0)
    assert len(noised) == 2


def test_estimate_privacy_cost_returns_dict():
    result = estimate_privacy_cost(1.1, 3, 10, 1, 1e-5)
    assert "epsilon_approx" in result
    assert "delta" in result
    assert "status" in result


def test_get_dp_info():
    info = get_dp_info()
    assert info["status"] == "DEMONSTRATION"
    assert "epsilon" in info
    assert "delta" in info


def test_simulate_mask_generation():
    mask = simulate_mask_generation("client_1", (4, 4))
    assert mask.shape == (4, 4)
    assert mask.dtype == np.float32


def test_simulate_secure_aggregate():
    updates = [np.ones((4,), dtype=np.float32) * i for i in range(3)]
    client_ids = ["client_1", "client_2", "client_3"]
    result = simulate_secure_aggregate(updates, client_ids)
    assert len(result) == 1


def test_get_secure_agg_info():
    info = get_secure_agg_info(enabled=False)
    assert info["status"] == "SIMULATION"
    assert info["enabled"] is False


def test_build_audit_report():
    report = build_audit_report(mode="DEMO")
    assert report.raw_data_uploaded_bytes == 0
    assert report.raw_dataset_transmitted is False
    assert report.local_training is True
    assert report.adapter_transmitted is True
    assert report.mode == "DEMO"


def test_get_privacy_boundary_summary():
    summary = get_privacy_boundary_summary()
    assert summary["raw_data_uploaded"] == "0 bytes"
    assert summary["raw_dataset_transmitted"] == "NO"
    assert summary["local_training"] != ""


def test_save_audit_report():
    report = build_audit_report(mode="DEMO")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_audit_report(report, tmpdir)
        assert os.path.exists(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["raw_data_uploaded_bytes"] == 0

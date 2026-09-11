import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from data.synthetic_data import (
    generate_client_dataset,
    get_all_clients_data,
    get_client_profile,
    CLIENT_PROFILES,
    _DOMAIN_SAMPLES,
)
from data.dataset import LocalDataset
from data.preprocessing import format_instruction_prompt, truncate_text


def test_client_profiles_count():
    assert len(CLIENT_PROFILES) == 3


def test_client_profile_fields():
    for profile in CLIENT_PROFILES:
        assert "client_id" in profile
        assert "domain" in profile
        assert "language" in profile
        assert "description" in profile


def test_get_client_profile_known():
    profile = get_client_profile("client_1")
    assert profile["domain"] == "Healthcare"
    assert profile["language"] == "Hindi"


def test_get_client_profile_unknown_raises():
    from core.exceptions import DataGenerationError
    with pytest.raises(DataGenerationError):
        get_client_profile("client_999")


def test_get_all_clients_data_returns_three():
    all_data = get_all_clients_data()
    assert len(all_data) == 3


def test_all_clients_have_samples():
    all_data = get_all_clients_data()
    for client_data in all_data:
        assert client_data["num_samples"] > 0
        assert len(client_data["samples"]) > 0


def test_samples_have_input_output():
    all_data = get_all_clients_data()
    for client_data in all_data:
        for sample in client_data["samples"]:
            assert "input" in sample
            assert "output" in sample
            assert len(sample["input"]) > 0
            assert len(sample["output"]) > 0


def test_generate_client_dataset_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_client_dataset("client_1", output_dir=tmpdir)
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["client_id"] == "client_1"
        assert data["domain"] == "Healthcare"
        assert len(data["samples"]) > 0


def test_generate_client_dataset_invalid_raises():
    from core.exceptions import DataGenerationError
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(DataGenerationError):
            generate_client_dataset("client_999", output_dir=tmpdir)


def test_local_dataset_from_dict():
    all_data = get_all_clients_data()
    ds = LocalDataset.from_dict(all_data[0])
    assert ds.client_id == "client_1"
    assert len(ds) > 0
    assert ds.domain == "Healthcare"
    assert ds.language == "Hindi"


def test_local_dataset_iteration():
    all_data = get_all_clients_data()
    ds = LocalDataset.from_dict(all_data[0])
    samples = list(ds)
    assert len(samples) == len(ds)


def test_local_dataset_prompts_and_responses():
    all_data = get_all_clients_data()
    ds = LocalDataset.from_dict(all_data[1])
    prompts = ds.get_prompts()
    responses = ds.get_responses()
    assert len(prompts) == len(responses) == len(ds)


def test_local_dataset_instruction_pairs():
    all_data = get_all_clients_data()
    ds = LocalDataset.from_dict(all_data[2])
    pairs = ds.as_instruction_pairs()
    assert len(pairs) == len(ds)
    for pair in pairs:
        assert "### Input:" in pair
        assert "### Response:" in pair


def test_format_instruction_prompt_with_response():
    result = format_instruction_prompt("What is 2+2?", "4")
    assert "### Input:" in result
    assert "### Response:" in result
    assert "4" in result


def test_format_instruction_prompt_without_response():
    result = format_instruction_prompt("What is ML?")
    assert "### Input:" in result
    assert "### Response:" in result
    assert result.endswith("### Response:\n")


def test_truncate_text_short():
    text = "Hello world"
    assert truncate_text(text, max_chars=100) == text


def test_truncate_text_long():
    text = "A" * 600
    result = truncate_text(text, max_chars=512)
    assert len(result) < 600
    assert result.endswith("...")

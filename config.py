from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    app_name: str = "PRIVFEDQLORA"
    app_version: str = "0.1.0"

    model_mode: str = field(default_factory=lambda: os.getenv("MODEL_MODE", "lightweight"))
    local_model_id: str = field(default_factory=lambda: os.getenv("LOCAL_MODEL_ID", "sshleifer/tiny-gpt2"))
    cloud_model_id: str = field(default_factory=lambda: os.getenv("CLOUD_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))

    lora_r: int = field(default_factory=lambda: int(os.getenv("LORA_R", "4")))
    lora_alpha: int = field(default_factory=lambda: int(os.getenv("LORA_ALPHA", "16")))
    lora_dropout: float = field(default_factory=lambda: float(os.getenv("LORA_DROPOUT", "0.05")))
    lora_target_modules: list = field(default_factory=lambda: ["c_attn", "c_proj"])

    local_batch_size: int = field(default_factory=lambda: int(os.getenv("LOCAL_BATCH_SIZE", "1")))
    max_seq_length: int = field(default_factory=lambda: int(os.getenv("MAX_SEQ_LENGTH", "64")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "64")))
    local_workers: int = field(default_factory=lambda: int(os.getenv("LOCAL_WORKERS", "1")))

    federated_rounds: int = field(default_factory=lambda: int(os.getenv("FEDERATED_ROUNDS", "1")))
    num_clients: int = 3

    ram_threshold_gb: float = 2.5
    storage_root: str = field(default_factory=lambda: os.getenv("STORAGE_ROOT", "storage"))
    adapters_dir: str = field(default_factory=lambda: os.path.join(os.getenv("STORAGE_ROOT", "storage"), "adapters"))
    datasets_dir: str = field(default_factory=lambda: os.path.join(os.getenv("STORAGE_ROOT", "storage"), "datasets"))
    results_dir: str = field(default_factory=lambda: os.path.join(os.getenv("STORAGE_ROOT", "storage"), "results"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.getenv("STORAGE_ROOT", "storage"), "logs"))

    dp_epsilon: float = field(default_factory=lambda: float(os.getenv("DP_EPSILON", "1.0")))
    dp_delta: float = field(default_factory=lambda: float(os.getenv("DP_DELTA", "1e-5")))
    dp_max_grad_norm: float = field(default_factory=lambda: float(os.getenv("DP_MAX_GRAD_NORM", "1.0")))
    dp_noise_multiplier: float = field(default_factory=lambda: float(os.getenv("DP_NOISE_MULTIPLIER", "1.1")))
    dp_enabled: bool = field(default_factory=lambda: os.getenv("DP_ENABLED", "true").lower() == "true")

    secure_agg_enabled: bool = field(default_factory=lambda: os.getenv("SECURE_AGG_ENABLED", "false").lower() == "true")

    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))


def get_config() -> AppConfig:
    return AppConfig()

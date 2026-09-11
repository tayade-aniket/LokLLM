from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _setup_sys_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_sys_path()


def _run_flower_simulation(num_rounds: int, cfg, output_dir: str) -> dict:
    import flwr as fl
    import torch
    from model.lora import build_bnb_config, build_lora_config
    from federated.client import build_flower_client
    from federated.strategy import build_fedavg_strategy
    from federated.server import build_flower_server_config
    from data.synthetic_data import get_all_clients_data
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.adapter import save_adapter_metadata

    model_id = cfg.cloud_model_id
    print(f"  Loading model: {model_id}")

    bnb_config = build_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=cfg.hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=cfg.hf_token or None,
    )

    lora_config = build_lora_config()
    all_data = get_all_clients_data()

    client_ids = ["client_1", "client_2", "client_3"]
    datasets = {d["client_id"]: d for d in all_data}

    def client_fn(cid: str):
        dataset = datasets.get(cid)
        return build_flower_client(cid, model, tokenizer, dataset)

    strategy = build_fedavg_strategy(min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3)

    start = time.perf_counter()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=build_flower_server_config(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.33},
    )
    duration = round(time.perf_counter() - start, 2)

    result_path = os.path.join(output_dir, "federated_results.json")
    result = {
        "num_rounds": num_rounds,
        "duration_seconds": duration,
        "losses_distributed": history.losses_distributed,
        "metrics_distributed": history.metrics_distributed,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PRIVFEDQLORA — Cloud Federated Simulation")
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="storage/results")
    args = parser.parse_args()

    from config import get_config
    cfg = get_config()

    print(f"[cloud/run_federated.py] Starting Flower federated simulation")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Clients: 3 (Healthcare/Hindi, Education/Marathi, Financial/Tamil)")
    print(f"  Strategy: FedAvg")

    try:
        import torch
        if not torch.cuda.is_available():
            print("[ERROR] CUDA is not available. This script requires a GPU environment.")
            sys.exit(1)
    except ImportError:
        print("[ERROR] PyTorch not installed.")
        sys.exit(1)

    try:
        import flwr
    except ImportError:
        print("[ERROR] Flower (flwr) not installed.")
        sys.exit(1)

    try:
        result = _run_flower_simulation(args.num_rounds, cfg, args.output_dir)
        print("[OK] Federated simulation complete")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[ERROR] Federated simulation failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

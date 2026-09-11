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


def _load_bnb_model(model_id: str, cfg):
    from model.lora import build_bnb_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return model, tokenizer


def _train_client(
    client_id: str,
    model,
    tokenizer,
    dataset,
    lora_config,
    cfg,
    output_dir: str,
) -> dict:
    import torch
    from peft import get_peft_model
    from torch.utils.data import DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length):
            self.items = []
            for s in samples:
                text = f"### Input:\n{s['input']}\n\n### Response:\n{s['output']}"
                enc = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
                self.items.append({k: v.squeeze(0) for k, v in enc.items()})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            return {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"], "labels": item["input_ids"]}

    peft_model = get_peft_model(model, lora_config)
    peft_model.train()

    ds = SimpleDataset(dataset["samples"], tokenizer, cfg.max_seq_length)
    loader = DataLoader(ds, batch_size=cfg.local_batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=3e-4)

    start = time.perf_counter()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(peft_model.device)
        attention_mask = batch["attention_mask"].to(peft_model.device)
        labels = batch["labels"].to(peft_model.device)

        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    duration = time.perf_counter() - start

    adapter_dir = os.path.join(output_dir, client_id)
    peft_model.save_pretrained(adapter_dir)

    return {
        "client_id": client_id,
        "avg_loss": total_loss / max(steps, 1),
        "steps": steps,
        "duration_seconds": round(duration, 2),
        "adapter_dir": adapter_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PRIVFEDQLORA — Cloud QLoRA Training Script")
    parser.add_argument("--model-id", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--client-id", type=str, default="client_1", help="Client to train (client_1/2/3)")
    parser.add_argument("--output-dir", type=str, default="storage/adapters", help="Output directory for adapter")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    args = parser.parse_args()

    from config import get_config
    cfg = get_config()

    model_id = args.model_id or cfg.cloud_model_id
    print(f"[cloud/train_qlora.py] Starting QLoRA training")
    print(f"  Model: {model_id}")
    print(f"  Client: {args.client_id}")
    print(f"  Output: {args.output_dir}")

    try:
        import torch
        if not torch.cuda.is_available():
            print("[ERROR] CUDA is not available. This script requires a GPU environment.")
            sys.exit(1)
    except ImportError:
        print("[ERROR] PyTorch not installed.")
        sys.exit(1)

    try:
        from model.lora import build_lora_config
        lora_config = build_lora_config(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=None,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to build LoRA config: {exc}")
        sys.exit(1)

    try:
        print(f"Loading model: {model_id}")
        model, tokenizer = _load_bnb_model(model_id, cfg)
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        sys.exit(1)

    try:
        from data.synthetic_data import get_all_clients_data
        all_data = get_all_clients_data()
        dataset = next((d for d in all_data if d["client_id"] == args.client_id), None)
        if dataset is None:
            print(f"[ERROR] Unknown client_id: {args.client_id}")
            sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        sys.exit(1)

    try:
        result = _train_client(args.client_id, model, tokenizer, dataset, lora_config, cfg, args.output_dir)
        print(f"[OK] Training complete.")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

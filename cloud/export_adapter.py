from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def _setup_sys_path() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_sys_path()


def _export_peft_adapter(model_id: str, adapter_dir: str, output_dir: str, cfg) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model.lora import build_bnb_config
    from model.adapter import save_adapter_metadata

    print(f"  Loading base model: {model_id}")
    bnb_config = build_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=cfg.hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=cfg.hf_token or None,
    )

    print(f"  Loading adapter from: {adapter_dir}")
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, adapter_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged adapter to: {output_dir}")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    files = list(Path(output_dir).rglob("*"))
    total_size_kb = sum(f.stat().st_size for f in files if f.is_file()) / 1024

    metadata = {
        "base_model": model_id,
        "adapter_source": adapter_dir,
        "export_dir": output_dir,
        "total_size_kb": round(total_size_kb, 2),
        "file_count": len([f for f in files if f.is_file()]),
        "export_type": "peft_adapter",
    }
    save_adapter_metadata(output_dir, metadata)
    return metadata


def _export_demo_adapter(output_dir: str) -> dict:
    from model.adapter import create_demo_adapter_record
    adapter_dir = create_demo_adapter_record(output_dir, round_num=1)
    print(f"  [DEMO] Created demo adapter record at: {adapter_dir}")
    return {"adapter_dir": adapter_dir, "mode": "DEMO"}


def main() -> None:
    parser = argparse.ArgumentParser(description="PRIVFEDQLORA — Adapter Export Script")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Source adapter directory from training")
    parser.add_argument("--output-dir", type=str, default="storage/adapters/exported", help="Export destination")
    parser.add_argument("--model-id", type=str, default=None, help="Base model ID")
    parser.add_argument("--demo", action="store_true", help="Create a demo adapter record (no GPU needed)")
    args = parser.parse_args()

    from config import get_config
    cfg = get_config()

    print(f"[cloud/export_adapter.py] Adapter Export")

    if args.demo:
        result = _export_demo_adapter(args.output_dir)
        print("[OK] Demo adapter record created")
        print(json.dumps(result, indent=2))
        return

    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available. Attempting CPU export (may be slow).")
    except ImportError:
        print("[ERROR] PyTorch not installed.")
        sys.exit(1)

    model_id = args.model_id or cfg.cloud_model_id
    adapter_dir = args.adapter_dir

    if not adapter_dir:
        print("[ERROR] --adapter-dir is required for real export. Use --demo for demo mode.")
        sys.exit(1)

    if not Path(adapter_dir).exists():
        print(f"[ERROR] Adapter directory not found: {adapter_dir}")
        sys.exit(1)

    try:
        result = _export_peft_adapter(model_id, adapter_dir, args.output_dir, cfg)
        print("[OK] Adapter exported")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[ERROR] Export failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

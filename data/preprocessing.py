from __future__ import annotations
from typing import Any, Optional


MAX_SEQ_LENGTH = 64


def format_instruction_prompt(input_text: str, response: str = "") -> str:
    if response:
        return f"### Input:\n{input_text}\n\n### Response:\n{response}"
    return f"### Input:\n{input_text}\n\n### Response:\n"


def truncate_text(text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def tokenize_sample(
    sample: dict,
    tokenizer: Any,
    max_length: int = MAX_SEQ_LENGTH,
) -> Optional[dict]:
    if tokenizer is None:
        return None

    text = format_instruction_prompt(sample.get("input", ""), sample.get("output", ""))

    try:
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
    except Exception:
        return None


def batch_tokenize(
    samples: list[dict],
    tokenizer: Any,
    max_length: int = MAX_SEQ_LENGTH,
    batch_size: int = 1,
) -> list[dict]:
    results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        for sample in batch:
            encoded = tokenize_sample(sample, tokenizer, max_length)
            if encoded is not None:
                results.append(encoded)
    return results

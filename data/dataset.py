from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterator, Optional

from core.exceptions import DataGenerationError


class LocalDataset:
    def __init__(self, client_id: str, data: dict[str, Any]):
        self._client_id = client_id
        self._data = data

    @classmethod
    def from_file(cls, path: str) -> "LocalDataset":
        p = Path(path)
        if not p.exists():
            raise DataGenerationError(f"Dataset file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(client_id=data.get("client_id", "unknown"), data=data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalDataset":
        return cls(client_id=data.get("client_id", "unknown"), data=data)

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def domain(self) -> str:
        return self._data.get("domain", "Unknown")

    @property
    def language(self) -> str:
        return self._data.get("language", "Unknown")

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    @property
    def samples(self) -> list[dict]:
        return self._data.get("samples", [])

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.samples)

    def get_prompts(self) -> list[str]:
        return [s["input"] for s in self.samples]

    def get_responses(self) -> list[str]:
        return [s["output"] for s in self.samples]

    def as_instruction_pairs(self) -> list[str]:
        pairs = []
        for s in self.samples:
            pairs.append(f"### Input:\n{s['input']}\n\n### Response:\n{s['output']}")
        return pairs

    def summary(self) -> dict:
        return {
            "client_id": self._client_id,
            "domain": self.domain,
            "language": self.language,
            "num_samples": len(self),
        }

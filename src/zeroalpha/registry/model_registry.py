"""Minimal model artifact registry."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json


def save_metadata(root: Path, model_version: str, metadata: dict[str, Any]) -> Path:
    path = root / "models" / model_version / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: asdict(value) if is_dataclass(value) else value for key, value in metadata.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path

"""Storage interfaces for raw and normalized datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Any
import json


class StorageUnavailableError(RuntimeError):
    pass


def partition_path(root: Path, table: str, **parts: str) -> Path:
    path = root / table
    for key, value in parts.items():
        path = path / f"{key}={value}"
    return path


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any] | object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)  # type: ignore[arg-type]
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
            count += 1
    return count


def write_parquet(path: Path, rows: list[Mapping[str, Any]]) -> int:
    """Write Parquet when pyarrow is installed.

    The core repo does not require pyarrow so tests and safety logic can run before
    heavy data dependencies are installed.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise StorageUnavailableError("install zeroalpha[data] to write Parquet") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return len(rows)

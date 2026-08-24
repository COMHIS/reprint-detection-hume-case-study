"""Reading, writing and provenance helpers.

Every stage writes its output to a new file and records the SHA-256 of its
inputs. No stage overwrites an earlier one; corrections are stored as overlays
that carry the original record, the new decision and the reason.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, prefix: str, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def manifest(
    stage: str,
    inputs: dict[str, str | Path],
    outputs: dict[str, str | Path],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance record for one stage: what went in, what came out, and how."""
    return {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters or {},
        "input_sha256": {
            name: sha256(path) for name, path in inputs.items() if Path(path).exists()
        },
        "output_sha256": {
            name: sha256(path) for name, path in outputs.items() if Path(path).exists()
        },
    }

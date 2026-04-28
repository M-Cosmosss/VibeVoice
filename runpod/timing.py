"""Structured [TIMING] logging for benchmark-friendly output."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any


_GPU_NAME: str | None = None


def gpu_name() -> str:
    global _GPU_NAME
    if _GPU_NAME is not None:
        return _GPU_NAME
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_NAME = torch.cuda.get_device_name(0)
        else:
            _GPU_NAME = "cpu"
    except Exception:
        _GPU_NAME = os.environ.get("RUNPOD_GPU_NAME", "unknown")
    return _GPU_NAME


def emit(stage: str, *, job_id: str | None = None, **fields: Any) -> None:
    parts = [f"stage={stage}"]
    if job_id is not None:
        parts.append(f"job={job_id}")
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    line = "[TIMING] " + " ".join(parts)
    print(line, file=sys.stdout, flush=True)


@contextmanager
def timed(stage: str, *, job_id: str | None = None, **fields: Any):
    """Context manager: emits a [TIMING] line on exit with duration_s.

    Usage:
        with timed("download", job_id=jid, url=u) as rec:
            ...
            rec["size_mb"] = 12.3   # extra fields appended at exit
    """
    extras: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        yield extras
    finally:
        dt = time.perf_counter() - t0
        emit(stage, job_id=job_id, duration_s=dt, **fields, **extras)

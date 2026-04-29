"""RunPod serverless handler — receive audio_url, return full transcript."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import traceback
from typing import Any

# Ensure /app is importable so `runpod_app.*` resolves regardless of CWD.
sys.path.insert(0, "/app")

import runpod  # type: ignore

from runpod_app.pipeline import run_job
from runpod_app.timing import emit


DEFAULT_CHUNK_MINUTES = int(os.environ.get("DEFAULT_CHUNK_MINUTES", "30"))
DEFAULT_CONCURRENCY = int(os.environ.get("DEFAULT_CONCURRENCY", "12"))


def _normalize_hotwords(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return ",".join(str(x).strip() for x in v if str(x).strip())
    s = str(v).strip()
    return s or None


async def _handle(event: dict[str, Any]) -> dict[str, Any]:
    inp = event.get("input") or {}
    audio_url = inp.get("audio_url")
    if not audio_url or not isinstance(audio_url, str):
        return {"error": "input.audio_url (string) is required"}

    chunk_minutes = int(inp.get("chunk_minutes") or DEFAULT_CHUNK_MINUTES)
    concurrency = int(inp.get("concurrency") or DEFAULT_CONCURRENCY)
    hotwords = _normalize_hotwords(inp.get("hotwords"))
    job_id = inp.get("job_id")
    if job_id is not None and not isinstance(job_id, str):
        return {"error": "input.job_id must be a string if provided"}

    try:
        result = await run_job(
            audio_url,
            hotwords=hotwords,
            chunk_minutes=chunk_minutes,
            concurrency=concurrency,
            job_id=job_id,
        )
    except Exception as e:
        emit("job_error", error=type(e).__name__)
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}

    return {
        "job_id": result.job_id,
        "text": result.text,
        "segments": result.segments,
        "files": result.files,
        "timing": result.timing,
    }


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_handle(event))

    result: dict[str, Any] | None = None
    error: BaseException | None = None

    def run_in_thread() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(_handle(event))
        except BaseException as e:
            error = e

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error
    if result is None:
        raise RuntimeError("handler completed without a result")
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

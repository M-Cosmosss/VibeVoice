"""End-to-end pipeline: download → split → concurrent ASR → merge."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from runpod_app.timing import emit, gpu_name, timed


VLLM_URL = f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '8000')}/v1/chat/completions"
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "/runpod-volume/transcripts"))
ASR_REQUEST_TIMEOUT_S = float(os.environ.get("ASR_REQUEST_TIMEOUT_S", "1800"))


SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)


@dataclass
class ChunkResult:
    index: int
    start_s: float
    end_s: float
    duration_s: float
    raw_content: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    parse_ok: bool = False
    asr_duration_s: float = 0.0


# ---------- audio io ----------

async def download_audio(url: str, dest_dir: Path, *, job_id: str) -> Path:
    """Stream-download audio URL to dest_dir. Filename derived from URL or random."""
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    out = dest_dir / f"input{suffix}"
    with timed("download", job_id=job_id, url=_truncate(url, 80)) as rec:
        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                size = 0
                with out.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1 << 20):
                        f.write(chunk)
                        size += len(chunk)
        rec["size_mb"] = round(size / (1 << 20), 2)
    return out


def probe_duration_s(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        stderr=subprocess.STDOUT,
    ).decode().strip()
    return float(out)


def split_audio(src: Path, chunk_seconds: int, work_dir: Path, *, job_id: str,
                duration_s: float) -> list[tuple[int, float, float, Path]]:
    """Split audio into chunk_seconds slices. Returns [(idx, start_s, end_s, path)].

    Re-encodes to 16kHz mono mp3 to keep payload small and decoder-friendly.
    """
    n = max(1, int((duration_s + chunk_seconds - 1) // chunk_seconds))
    chunks: list[tuple[int, float, float, Path]] = []
    with timed("split", job_id=job_id, num_chunks=n, chunk_seconds=chunk_seconds):
        for i in range(n):
            start = i * chunk_seconds
            end = min((i + 1) * chunk_seconds, duration_s)
            out_path = work_dir / f"chunk_{i:03d}.mp3"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start}", "-t", f"{end - start}",
                "-i", str(src),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "libmp3lame", "-q:a", "4",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
            chunks.append((i, start, end, out_path))
    return chunks


# ---------- ASR ----------

def _build_payload(audio_b64: str, mime: str, duration_s: float,
                   hotwords: str | None) -> dict[str, Any]:
    if hotwords and hotwords.strip():
        prompt = (
            f"This is a {duration_s:.2f} seconds audio, with extra info: "
            f"{hotwords.strip()}\n\n"
            f"Please transcribe it with these keys: " + ", ".join(SHOW_KEYS)
        )
    else:
        prompt = (
            f"This is a {duration_s:.2f} seconds audio, please transcribe it "
            f"with these keys: " + ", ".join(SHOW_KEYS)
        )
    data_url = f"data:{mime};base64,{audio_b64}"
    return {
        "model": "vibevoice",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "max_tokens": 32768,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }


async def asr_chunk(client: httpx.AsyncClient, idx: int, start_s: float,
                    end_s: float, path: Path, hotwords: str | None,
                    *, job_id: str) -> ChunkResult:
    duration = end_s - start_s
    mime = "audio/mpeg"
    audio_b64 = base64.b64encode(path.read_bytes()).decode()
    payload = _build_payload(audio_b64, mime, duration, hotwords)

    emit("asr_chunk_start", job_id=job_id, chunk=idx, start=start_s, end=end_s)
    t0 = time.perf_counter()
    resp = await client.post(VLLM_URL, json=payload, timeout=ASR_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    dt = time.perf_counter() - t0
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    segments, parse_ok = _parse_segments(content, offset_s=start_s)
    emit("asr_chunk_done", job_id=job_id, chunk=idx, duration_s=dt,
         chunk_audio_s=duration, rtf=dt / max(duration, 1e-6),
         segments=len(segments), parse_ok=int(parse_ok))
    return ChunkResult(
        index=idx, start_s=start_s, end_s=end_s, duration_s=duration,
        raw_content=content, segments=segments, parse_ok=parse_ok,
        asr_duration_s=dt,
    )


async def run_chunks(chunks: list[tuple[int, float, float, Path]],
                     hotwords: str | None, concurrency: int,
                     *, job_id: str) -> list[ChunkResult]:
    sem = asyncio.Semaphore(max(1, concurrency))
    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency * 2)

    async def _bound(client, *args):
        async with sem:
            return await asr_chunk(client, *args, hotwords=hotwords, job_id=job_id)

    async with httpx.AsyncClient(limits=limits, timeout=ASR_REQUEST_TIMEOUT_S) as client:
        with timed("asr_total", job_id=job_id, concurrency=concurrency,
                   num_chunks=len(chunks)):
            results = await asyncio.gather(*[
                _bound(client, idx, s, e, p) for idx, s, e, p in chunks
            ])
    results.sort(key=lambda r: r.index)
    return results


# ---------- merge / parsing ----------

_TIME_RE = re.compile(
    r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?$"
)


def _parse_time_to_s(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    m = _TIME_RE.match(s)
    if m:
        h = int(m.group(1) or 0)
        mn = int(m.group(2))
        sec = int(m.group(3))
        ms = int((m.group(4) or "0").ljust(3, "0")[:3])
        return h * 3600 + mn * 60 + sec + ms / 1000.0
    try:
        return float(s)
    except ValueError:
        return None


def _extract_json(text: str) -> Any | None:
    """Extract a JSON value from possibly fenced text."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # try to find the first [...] or {...}
    for opener, closer in [("[", "]"), ("{", "}")]:
        i = s.find(opener)
        j = s.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def _parse_segments(content: str, *, offset_s: float) -> tuple[list[dict[str, Any]], bool]:
    """Parse model output into normalized segments, applying global offset."""
    parsed = _extract_json(content)
    if parsed is None:
        return [], False
    if isinstance(parsed, dict):
        for k in ("segments", "results", "data", "transcription"):
            if isinstance(parsed.get(k), list):
                parsed = parsed[k]
                break
    if not isinstance(parsed, list):
        return [], False

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        st = _parse_time_to_s(
            item.get("Start time") or item.get("start") or item.get("start_time")
        )
        ed = _parse_time_to_s(
            item.get("End time") or item.get("end") or item.get("end_time")
        )
        spk = item.get("Speaker ID") or item.get("speaker") or item.get("speaker_id")
        text = item.get("Content") or item.get("text") or item.get("content") or ""
        seg = {
            "start": (st + offset_s) if st is not None else None,
            "end": (ed + offset_s) if ed is not None else None,
            "speaker": str(spk) if spk is not None else None,
            "text": str(text).strip(),
        }
        if seg["text"]:
            out.append(seg)
    return out, True


def merge_results(results: list[ChunkResult]) -> tuple[str, list[dict[str, Any]]]:
    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for r in results:
        if r.segments:
            segments.extend(r.segments)
            text_parts.append("\n".join(s["text"] for s in r.segments if s["text"]))
        else:
            # parse failed: keep raw content with marker so nothing is lost
            text_parts.append(
                f"[chunk {r.index} raw — JSON parse failed]\n{r.raw_content.strip()}"
            )
    return "\n".join(p for p in text_parts if p), segments


# ---------- public entry ----------

@dataclass
class JobResult:
    job_id: str
    text: str
    segments: list[dict[str, Any]]
    files: dict[str, str]
    timing: dict[str, Any]


async def run_job(audio_url: str, *, hotwords: str | None,
                  chunk_minutes: int, concurrency: int,
                  job_id: str | None = None) -> JobResult:
    job_id = job_id or uuid.uuid4().hex[:12]
    chunk_seconds = max(60, int(chunk_minutes * 60))
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    emit("job_start", job_id=job_id, chunk_minutes=chunk_minutes,
         concurrency=concurrency, gpu=gpu_name())
    job_t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"vv_{job_id}_") as tmp:
        work = Path(tmp)
        audio_path = await download_audio(audio_url, work, job_id=job_id)

        with timed("probe", job_id=job_id) as rec:
            duration_s = probe_duration_s(audio_path)
            rec["audio_duration_s"] = round(duration_s, 2)

        chunks = split_audio(audio_path, chunk_seconds, work,
                             job_id=job_id, duration_s=duration_s)
        results = await run_chunks(chunks, hotwords, concurrency, job_id=job_id)

        with timed("merge", job_id=job_id) as rec:
            text, segments = merge_results(results)
            rec["segments"] = len(segments)
            rec["chars"] = len(text)

    json_path = TRANSCRIPT_DIR / f"{job_id}.json"
    txt_path = TRANSCRIPT_DIR / f"{job_id}.txt"
    total_s = time.perf_counter() - job_t0
    timing = {
        "total_s": round(total_s, 3),
        "audio_duration_s": round(duration_s, 2),
        "rtf": round(total_s / max(duration_s, 1e-6), 4),
        "num_chunks": len(chunks),
        "chunk_seconds": chunk_seconds,
        "concurrency": concurrency,
        "asr_per_chunk_s": [round(r.asr_duration_s, 3) for r in results],
        "asr_total_s": round(sum(r.asr_duration_s for r in results), 3),
        "asr_max_s": round(max((r.asr_duration_s for r in results), default=0.0), 3),
        "gpu": gpu_name(),
    }
    payload = {
        "job_id": job_id,
        "audio_url": audio_url,
        "text": text,
        "segments": segments,
        "timing": timing,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    txt_path.write_text(text)

    emit("job_done", job_id=job_id, total_s=total_s,
         audio_s=duration_s, rtf=timing["rtf"], gpu=gpu_name(),
         num_chunks=len(chunks), concurrency=concurrency)

    return JobResult(
        job_id=job_id, text=text, segments=segments,
        files={"json": str(json_path), "txt": str(txt_path)},
        timing=timing,
    )


# ---------- helpers ----------

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."

# VibeVoice ASR — RunPod Serverless image.
# Base: official vLLM image (verified compatible per docs/vibevoice-vllm-asr.md).
FROM vllm/vllm-openai:v0.14.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_PORT=8000 \
    TRANSCRIPT_DIR=/tmp/transcripts \
    MODEL_ID=microsoft/VibeVoice-ASR \
    HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
    TOKENIZER_PATCH_DIR=/opt/vibevoice-tokenizer \
    MODEL_PATH=/tmp/vibevoice-asr-runtime

# ---- 1. apt deps (rare changes; cached) ------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# ---- 2. hf-transfer (kept for small build-time HF downloads) ----------------
RUN pip install --no-cache-dir hf-transfer==0.1.8

# ---- 3. Project source + editable install (changes when vibevoice/ updates) -
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY vibevoice /app/vibevoice
COPY vllm_plugin /app/vllm_plugin
RUN pip install --no-cache-dir "/app[vllm]" \
    && pip install --no-cache-dir runpod==1.7.7 httpx==0.27.2

# ---- 4. Generate tokenizer files (small layer, no ASR weights baked in) -----
# Runtime links cached model weights from RunPod's model cache, then overlays
# these VibeVoice tokenizer files without mutating the cache.
RUN python3 -m vllm_plugin.tools.generate_tokenizer_files --output ${TOKENIZER_PATCH_DIR}

# ---- 5. Runtime entrypoint code (changes most often → last layer) ----------
COPY runpod/timing.py /app/runpod_app/timing.py
COPY runpod/pipeline.py /app/runpod_app/pipeline.py
COPY runpod/handler.py /app/runpod_app/handler.py
COPY runpod/start.sh /app/runpod_app/start.sh
RUN chmod +x /app/runpod_app/start.sh && touch /app/runpod_app/__init__.py

ENTRYPOINT ["/app/runpod_app/start.sh"]

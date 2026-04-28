# VibeVoice ASR — RunPod Serverless image
# Base: official vLLM image (verified compatible per docs/vibevoice-vllm-asr.md)
FROM vllm/vllm-openai:v0.14.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=microsoft/VibeVoice-ASR \
    VLLM_PORT=8000 \
    TRANSCRIPT_DIR=/runpod-volume/transcripts

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install VibeVoice (with vllm extras) and runtime deps
COPY pyproject.toml /app/pyproject.toml
COPY vibevoice /app/vibevoice
COPY vllm_plugin /app/vllm_plugin

RUN pip install --no-cache-dir "/app[vllm]" \
    && pip install --no-cache-dir runpod==1.7.7 httpx==0.27.2 hf-transfer==0.1.8

COPY runpod/timing.py /app/runpod_app/timing.py
COPY runpod/pipeline.py /app/runpod_app/pipeline.py
COPY runpod/handler.py /app/runpod_app/handler.py
COPY runpod/start.sh /app/runpod_app/start.sh
RUN chmod +x /app/runpod_app/start.sh && touch /app/runpod_app/__init__.py

ENTRYPOINT ["/app/runpod_app/start.sh"]

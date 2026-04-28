# VibeVoice ASR — RunPod Serverless image (model baked in, no Network Volume needed).
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
    MODEL_PATH=/opt/vibevoice-asr

# ---- 1. apt deps (rare changes; cached) ------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# ---- 2. hf-transfer (needed before model download for speed) ---------------
RUN pip install --no-cache-dir hf-transfer==0.1.8

# ---- 3. Bake model weights (~14 GB layer; cached unless MODEL_ID changes) --
# Placed BEFORE copying any project source so day-to-day code changes do NOT
# invalidate this expensive layer.
ARG MODEL_ID=microsoft/VibeVoice-ASR
RUN python3 - <<PY
from huggingface_hub import snapshot_download
import os
path = snapshot_download("${MODEL_ID}", local_dir=os.environ["MODEL_PATH"])
print(f"Model baked at: {path}")
PY

# ---- 4. Project source + editable install (changes when vibevoice/ updates) -
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY vibevoice /app/vibevoice
COPY vllm_plugin /app/vllm_plugin
RUN pip install --no-cache-dir "/app[vllm]" \
    && pip install --no-cache-dir runpod==1.7.7 httpx==0.27.2

# ---- 5. Generate tokenizer files (small, fast layer, depends on plugin) ----
RUN python3 -m vllm_plugin.tools.generate_tokenizer_files --output ${MODEL_PATH}

# ---- 6. Runtime entrypoint code (changes most often → last layer) ----------
COPY runpod/timing.py /app/runpod_app/timing.py
COPY runpod/pipeline.py /app/runpod_app/pipeline.py
COPY runpod/handler.py /app/runpod_app/handler.py
COPY runpod/start.sh /app/runpod_app/start.sh
RUN chmod +x /app/runpod_app/start.sh && touch /app/runpod_app/__init__.py

ENTRYPOINT ["/app/runpod_app/start.sh"]

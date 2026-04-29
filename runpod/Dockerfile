# VibeVoice ASR — RunPod Serverless image.
# Base: RunPod's vLLM worker image, which is more likely to be cached on RunPod hosts.
FROM runpod/worker-v1-vllm:v2.13.0

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

# ---- 1. ASR system deps (small custom layer) --------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# ---- 2. VibeVoice plugin only; keep vLLM/PyTorch from the RunPod base --------
WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY vibevoice /app/vibevoice
COPY vllm_plugin /app/vllm_plugin
RUN python3 -m pip install --no-cache-dir --no-deps /app \
    && python3 -m pip install --no-cache-dir runpod==1.7.7 httpx==0.27.2 hf-transfer==0.1.8

# ---- 3. Generate tokenizer files (small layer, no ASR weights baked in) -----
# Runtime links cached model weights from RunPod's model cache, then overlays
# these VibeVoice tokenizer files without mutating the cache.
# Run as a script (not `-m`) to skip vllm_plugin/__init__.py, which imports
# vibevoice and would collide with transformers' AutoModel registry on this base image.
RUN python3 /app/vllm_plugin/tools/generate_tokenizer_files.py --output ${TOKENIZER_PATCH_DIR} \
    && rm -rf /app/vibevoice /app/vllm_plugin /app/pyproject.toml /app/README.md

# ---- 4. Runtime entrypoint code (changes most often -> last layer) ----------
COPY runpod/timing.py /app/runpod_app/timing.py
COPY runpod/pipeline.py /app/runpod_app/pipeline.py
COPY runpod/handler.py /app/runpod_app/handler.py
COPY runpod/start.sh /app/runpod_app/start.sh
RUN chmod +x /app/runpod_app/start.sh && touch /app/runpod_app/__init__.py

ENTRYPOINT ["/app/runpod_app/start.sh"]

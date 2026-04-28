#!/usr/bin/env bash
# Entry: launch vLLM in background, then run RunPod handler.
# Model weights are baked into the image at $MODEL_PATH.
set -euo pipefail

log() { echo "[start] $*"; }

: "${MODEL_PATH:=/opt/vibevoice-asr}"
: "${VLLM_PORT:=8000}"
: "${TRANSCRIPT_DIR:=/tmp/transcripts}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_NUM_SEQS:=8}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${VLLM_READY_TIMEOUT_S:=300}"

mkdir -p "$TRANSCRIPT_DIR"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    log "ERROR: model not found at $MODEL_PATH (config.json missing). Image is broken."
    exit 1
fi
log "MODEL_PATH=$MODEL_PATH (baked-in)"

log "launching vLLM serve on port ${VLLM_PORT} (max_model_len=${MAX_MODEL_LEN}, max_num_seqs=${MAX_NUM_SEQS})"
VLLM_T0=$(date +%s)
vllm serve "$MODEL_PATH" \
    --served-model-name vibevoice \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --chat-template-content-format openai \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --port "$VLLM_PORT" \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
log "vLLM pid=$VLLM_PID"

log "waiting for vLLM /v1/models (timeout=${VLLM_READY_TIMEOUT_S}s)"
deadline=$(( $(date +%s) + VLLM_READY_TIMEOUT_S ))
until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM exited before ready. tail of log:"
        tail -n 200 /tmp/vllm.log || true
        exit 1
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
        log "ERROR: vLLM not ready within ${VLLM_READY_TIMEOUT_S}s. tail of log:"
        tail -n 200 /tmp/vllm.log || true
        exit 1
    fi
    sleep 2
done
VLLM_READY_S=$(( $(date +%s) - VLLM_T0 ))
log "[TIMING] stage=vllm_ready duration_s=${VLLM_READY_S}"

log "starting RunPod handler"
exec python3 -u /app/runpod_app/handler.py

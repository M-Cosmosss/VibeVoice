#!/usr/bin/env bash
# Entry: prepare model in Network Volume, launch vLLM in background, run RunPod handler.
set -euo pipefail

log() { echo "[start] $*"; }

: "${HF_HOME:=/runpod-volume/hf}"
: "${MODEL_ID:=microsoft/VibeVoice-ASR}"
: "${VLLM_PORT:=8000}"
: "${TRANSCRIPT_DIR:=/runpod-volume/transcripts}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_NUM_SEQS:=8}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${VLLM_READY_TIMEOUT_S:=900}"

mkdir -p "$HF_HOME" "$TRANSCRIPT_DIR"

# ---- Step 1: prefill model into Network Volume (single-flight via flock) ----
PREP_LOCK="${HF_HOME}/.vibevoice-prep.lock"
PREP_DONE="${HF_HOME}/.vibevoice-prep.done"

prepare_model() {
    log "preparing model id=${MODEL_ID} hf_home=${HF_HOME}"
    local t0; t0=$(date +%s)
    python3 - <<PY
import os, time
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(os.environ["MODEL_ID"])
print(f"[start] snapshot_download done path={path} duration_s={time.time()-t0:.2f}")
PY

    log "generating tokenizer files"
    local model_path
    model_path=$(python3 -c "from huggingface_hub import snapshot_download; import os; print(snapshot_download(os.environ['MODEL_ID']))")
    python3 -m vllm_plugin.tools.generate_tokenizer_files --output "$model_path"
    echo "$model_path" > "$PREP_DONE"
    log "model prepared in $(( $(date +%s) - t0 ))s, path=$model_path"
}

(
    flock -x 200
    if [[ ! -f "$PREP_DONE" ]]; then
        prepare_model
    else
        log "model already prepared (marker $PREP_DONE exists), skipping download"
    fi
) 200>"$PREP_LOCK"

MODEL_PATH=$(cat "$PREP_DONE")
log "MODEL_PATH=$MODEL_PATH"

# ---- Step 2: launch vLLM in background ----
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

# ---- Step 3: wait until vLLM ready ----
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

# ---- Step 4: hand over to RunPod handler ----
log "starting RunPod handler"
exec python3 -u /app/runpod_app/handler.py

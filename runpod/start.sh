#!/usr/bin/env bash
# Entry: launch vLLM in background, then run RunPod handler.
# Model weights are provided by RunPod's Hugging Face model cache.
set -euo pipefail

log() { echo "[start] $*"; }

: "${MODEL_ID:=${MODEL_NAME:-microsoft/VibeVoice-ASR}}"
: "${MODEL_REVISION:=}"
: "${HF_CACHE_ROOT:=/runpod-volume/huggingface-cache/hub}"
: "${TOKENIZER_PATCH_DIR:=/opt/vibevoice-tokenizer}"
: "${MODEL_PATH:=/tmp/vibevoice-asr-runtime}"
: "${VLLM_PORT:=8000}"
: "${TRANSCRIPT_DIR:=/tmp/transcripts}"
: "${MAX_MODEL_LEN:=32768}"
: "${MAX_NUM_SEQS:=8}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${VLLM_READY_TIMEOUT_S:=300}"

mkdir -p "$TRANSCRIPT_DIR"

CACHED_MODEL_PATH=$(
    python3 - <<'PY'
import os
import sys

model_id = os.environ["MODEL_ID"]
revision = os.environ.get("MODEL_REVISION", "").strip()
cache_root = os.environ["HF_CACHE_ROOT"]

if "/" not in model_id:
    raise SystemExit(f"MODEL_ID must be in org/name format, got: {model_id!r}")

org, name = model_id.split("/", 1)
expected_roots = [
    os.path.join(cache_root, f"models--{org}--{name}"),
    os.path.join(cache_root, f"models--{org.lower()}--{name.lower()}"),
]

model_root = next((p for p in expected_roots if os.path.isdir(p)), None)
if model_root is None and os.path.isdir(cache_root):
    wanted = f"models--{org}--{name}".lower()
    for entry in os.listdir(cache_root):
        if entry.lower() == wanted:
            model_root = os.path.join(cache_root, entry)
            break

if model_root is None:
    raise SystemExit(
        "RunPod cached model not found. Configure the endpoint Model field as "
        f"{model_id!r}; looked under {cache_root!r}."
    )

snapshots_dir = os.path.join(model_root, "snapshots")
candidates = []
if revision:
    candidates.append(os.path.join(snapshots_dir, revision))

refs_main = os.path.join(model_root, "refs", "main")
if os.path.isfile(refs_main):
    with open(refs_main, "r", encoding="utf-8") as f:
        ref = f.read().strip()
    if ref:
        candidates.append(os.path.join(snapshots_dir, ref))

if os.path.isdir(snapshots_dir):
    candidates.extend(
        os.path.join(snapshots_dir, d)
        for d in sorted(os.listdir(snapshots_dir))
        if os.path.isdir(os.path.join(snapshots_dir, d))
    )

for candidate in candidates:
    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
        print(candidate)
        break
else:
    raise SystemExit(f"No cached snapshot with config.json found under {snapshots_dir!r}.")
PY
)

log "cached model path=${CACHED_MODEL_PATH}"
log "preparing runtime model path=${MODEL_PATH}"
rm -rf "$MODEL_PATH"
mkdir -p "$MODEL_PATH"
shopt -s dotglob nullglob
for item in "$CACHED_MODEL_PATH"/*; do
    ln -s "$item" "$MODEL_PATH/$(basename "$item")"
done
shopt -u dotglob nullglob

if [[ ! -d "$TOKENIZER_PATCH_DIR" ]]; then
    log "ERROR: tokenizer patch directory not found at $TOKENIZER_PATCH_DIR"
    exit 1
fi
for file in vocab.json merges.txt tokenizer.json tokenizer_config.json added_tokens.json special_tokens_map.json; do
    rm -f "$MODEL_PATH/$file"
done
cp -f "$TOKENIZER_PATCH_DIR"/* "$MODEL_PATH"/

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    log "ERROR: model not found at $MODEL_PATH (config.json missing)."
    exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
log "MODEL_PATH=$MODEL_PATH (RunPod cached model + tokenizer overlay)"

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

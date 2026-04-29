# VibeVoice ASR — RunPod Serverless 部署

端到端 ASR 服务：接收 `audio_url` → 切片 → 并发转写 → 合并 → 落盘 → 返回完整文稿。

## 1. 架构

- **基础镜像**：`runpod/worker-v1-vllm:v2.13.0`，复用 RunPod 官方 vLLM worker 大层，减少 Serverless 冷启动拉镜像时间；旧版 `vllm/vllm-openai:v0.14.1` Dockerfile 备份在 `runpod/Dockerfile.vllm-openai-backup`。
- **模型权重**：使用 RunPod Serverless **Model cache**，worker 从 `/runpod-volume/huggingface-cache/hub` 解析本地 snapshot，不把 14GB 权重烤进镜像。
- **服务端做端到端编排**：单次请求完成下载、切片、并发 ASR、合并。结果写入 `/tmp/transcripts/{job_id}.{json,txt}` 并直接在 API 返回。
- **日志**：每阶段输出 `[TIMING] stage=... duration_s=...` 结构化日志，便于切换 GPU benchmark。

### 镜像分层（按变更频率排序，最大化 cache 命中）

```
Layer 1: runpod/worker-v1-vllm 官方基础镜像（RunPod 节点更可能命中缓存）
Layer 2: ffmpeg/libsndfile/curl            （小层）
Layer 3: vibevoice / vllm_plugin 源码       （--no-deps 安装，不重装 vLLM/PyTorch）
Layer 4: 生成 tokenizer patch 文件          （随 plugin 变，小层）
Layer 5: runpod/*.py 业务代码               （日常迭代只改这层）
```

`Layer 3` 使用 `pip install --no-deps /app`，所以运行时依赖需要在 `runpod/Dockerfile` 里显式维护。`audio_url` 请求路径依赖 vLLM audio support；镜像构建会 smoke check `librosa` / `soundfile`，避免缺 audio runtime 依赖的问题拖到线上请求才暴露。

日常改 `runpod/handler.py` / `runpod/pipeline.py` 只重建最后一层（~10 KB）。仓库根目录 `.dockerignore` 会排除 docs、demo、tests、训练数据和本地输出，减少 GitHub builder 的 build context。

## 2. 构建镜像

### 2.1 推荐：RunPod GitHub 集成（云端 builder，零本地流量）

- RunPod 控制台 → **Serverless** → **New Endpoint** → **GitHub** 来源
- Repo: `M-Cosmosss/VibeVoice`
- Branch: `main`
- Dockerfile path: `runpod/Dockerfile`
- Build context: `/`（仓库根目录）
- 后续 `git push` 自动重新构建 + 滚动更新

### 2.2 备选：本地构建（需要跨架构 + push）

```bash
cd /path/to/VibeVoice
docker buildx create --name xbuild --use --bootstrap
docker buildx build \
  --platform linux/amd64 \
  -f runpod/Dockerfile \
  -t <your-registry>/vibevoice-asr-runpod:latest \
  --push .
```

首次构建通常 15–25 分钟，增量构建通常几分钟。

## 3. RunPod Endpoint 配置

- **GPU**：L40S 48 GB（推荐起步）
- **Active Workers**：`0`（scale-to-zero）
- **Max Workers**：按客户端最大并发上限设（默认建议 4）
- **Idle Timeout**：5–10s
- **FlashBoot**：ON
- **Container Disk**：20–30 GB
- **Network Volume**：**不需要**
- **Model**：`microsoft/VibeVoice-ASR`（必须填写，用于启用 RunPod Model cache）
- **Execution Timeout**：根据最长音频估，参考下面的耗时表
- **Allow GPU fallback**：可选开（L40S 紧张时自动升档到 A100/H100，按使用计费）

### 环境变量（可选覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_MODEL_LEN` | `32768` | vLLM 上下文长度（30min 切片够用） |
| `MAX_NUM_SEQS` | `200` | vLLM 单 worker scheduler 上限；实际并发由 `concurrency` 控制 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM 显存占比 |
| `ENABLE_FP8` | `false` | 创建 Endpoint/Template 时的量化开关；设为 `true` 会启用 vLLM FP8 权重量化和 FP8 KV cache |
| `VLLM_QUANTIZATION` | 空 | 细粒度覆盖 vLLM `--quantization`；`ENABLE_FP8=true` 且未设置时为 `fp8` |
| `VLLM_KV_CACHE_DTYPE` | `auto` | 细粒度覆盖 vLLM `--kv-cache-dtype`；`ENABLE_FP8=true` 且未设置时为 `fp8` |
| `VLLM_CALCULATE_KV_SCALES` | `false` | 细粒度覆盖 vLLM `--calculate-kv-scales`；`ENABLE_FP8=true` 且未设置时为 `true` |
| `MODEL_ID` / `MODEL_NAME` | `microsoft/VibeVoice-ASR` | HF 模型 ID；RunPod Model 字段通常会注入 `MODEL_NAME` |
| `HF_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | RunPod cached model 根目录 |
| `MODEL_PATH` | `/tmp/vibevoice-asr-runtime` | 启动时创建的 runtime 模型目录，权重为 symlink，tokenizer 为镜像内 patch |
| `DEFAULT_CHUNK_MINUTES` | `30` | 切片长度（分钟） |
| `DEFAULT_CONCURRENCY` | `12` | 单请求内并发切片数 |
| `ASR_PROMPT_TOKEN_RESERVE` | `512` | 为 system/user prompt 预留的输入 token 预算；输出 token 自动使用 `MAX_MODEL_LEN` 剩余空间 |
| `VLLM_READY_TIMEOUT_S` | `300` | 等待 vLLM 就绪超时 |
| `ASR_REQUEST_TIMEOUT_S` | `1800` | 单切片 ASR HTTP 超时 |
| `TRANSCRIPT_DIR` | `/tmp/transcripts` | 文稿落盘目录 |

FP8 是 vLLM engine 启动期参数，不能作为单次 `/run` / `/runsync` 音频任务输入动态切换。要测试 FP8，请在创建或更新 RunPod Endpoint/Template 时添加环境变量 `ENABLE_FP8=true`，让 worker 冷启动时用 FP8 模式加载模型。

## 4. 调用

### 4.1 同步 vs 异步

| 端点 | 何时用 | 上限 |
|---|---|---|
| `POST /v2/<EID>/runsync` | 估算总耗时 < ~280s 的请求（≤ 4h 音频在 L40S 上够用） | 默认 **300s** |
| `POST /v2/<EID>/run` + 轮询 `/status/{id}` | 长音频或不确定时长 | worker execution timeout，可配几小时 |

按 L40S + concurrency=12 估算：

| 音频时长 | 切片数 | 预估总耗时 | 推荐 |
|---|---|---|---|
| 30 min | 1 | ~45 s | runsync |
| 1 h | 2 | ~50 s | runsync |
| 2 h | 4 | ~50 s | runsync |
| 4 h | 8 | ~100 s | runsync |
| 8 h | 16 | ~200 s | runsync 边缘，建议 run+poll |
| 16 h+ | 32+ | 400 s+ | run+poll |

### 4.2 输入

```json
{
  "input": {
    "audio_url": "https://example.com/podcast.mp3",
    "hotwords": ["VibeVoice", "微软"],
    "chunk_minutes": 30,
    "concurrency": 12,
    "job_id": "optional-custom-id"
  }
}
```

### 4.3 输出

```json
{
  "job_id": "abc123",
  "text": "完整合并后的文稿……",
  "segments": [
    {"start": 0.12, "end": 3.45, "speaker": "1", "text": "..."}
  ],
  "files": {
    "json": "/tmp/transcripts/abc123.json",
    "txt":  "/tmp/transcripts/abc123.txt"
  },
  "timing": {
    "summary": {
      "total_elapsed_time": {"seconds": 92.7, "readable": "1m32.700s"},
      "audio_duration": {"seconds": 3600.0, "readable": "1h0m0.000s"},
      "realtime_factor": 0.0258,
      "gpu": "NVIDIA L40S"
    },
    "configuration": {
      "chunk_count": 2,
      "chunk_duration": {"seconds": 1800.0, "readable": "30m0.000s"},
      "concurrency": 12,
      "quantization": {
        "enable_fp8": false,
        "quantization": "none",
        "kv_cache_dtype": "auto",
        "calculate_kv_scales": false
      }
    },
    "stages": {
      "download_audio": {"seconds": 4.21, "readable": "4.210s"},
      "probe_audio_duration": {"seconds": 0.08, "readable": "0.080s"},
      "split_audio": {"seconds": 1.1, "readable": "1.100s"},
      "transcribe_audio_wall_time": {"seconds": 87.3, "readable": "1m27.300s"},
      "merge_transcripts": {"seconds": 0.05, "readable": "0.050s"}
    },
    "chunk_summary": {
      "request_preparation_total": {"seconds": 1.26, "readable": "1.260s"},
      "request_preparation_slowest_chunk": {"seconds": 0.64, "readable": "0.640s"},
      "transcription_total_across_chunks": {"seconds": 87.35, "readable": "1m27.350s"},
      "transcription_slowest_chunk": {"seconds": 45.22, "readable": "45.220s"}
    },
    "chunks": [
      {
        "chunk_index": 0,
        "audio_start": {"seconds": 0.0, "readable": "0.000s"},
        "audio_end": {"seconds": 1800.0, "readable": "30m0.000s"},
        "audio_duration": {"seconds": 1800.0, "readable": "30m0.000s"},
        "request_preparation_time": {"seconds": 0.62, "readable": "0.620s"},
        "transcription_time": {"seconds": 42.13, "readable": "42.130s"},
        "processing_time": {"seconds": 42.75, "readable": "42.750s"},
        "realtime_factor": 0.0234,
        "segments": 180,
        "parsed_as_json": true
      }
    ]
  }
}
```

> 注意：`files` 路径在 worker 容器内，worker 销毁后会丢失。要保留历史请把 `text` / `segments` 写入你自己的存储（S3/OSS/DB）。

### 4.4 同步调用示例

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://.../foo.mp3", "concurrency": 12}}'
```

### 4.5 异步调用示例

```bash
# 1. 提交
JOB=$(curl -s -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://.../long.mp3"}}' | jq -r .id)

# 2. 轮询
while true; do
  STATUS=$(curl -s "https://api.runpod.ai/v2/<ENDPOINT_ID>/status/$JOB" \
    -H "Authorization: Bearer $RUNPOD_API_KEY")
  echo "$STATUS" | jq .status
  [[ $(echo "$STATUS" | jq -r .status) == "COMPLETED" ]] && echo "$STATUS" | jq .output && break
  sleep 5
done
```

## 5. 日志（benchmark 友好）

```
[TIMING] stage=vllm_ready duration_s=18
[TIMING] stage=job_start job=abc123 chunk_minutes=30 concurrency=12 gpu=NVIDIA L40S
[TIMING] stage=download job=abc123 url=https://... duration_s=4.210 size_mb=58.30
[TIMING] stage=probe job=abc123 duration_s=0.082 audio_duration_s=3600.00
[TIMING] stage=split job=abc123 duration_s=1.103 num_chunks=2 chunk_seconds=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=0 start=0 end=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=1 start=1800 end=3600
[TIMING] stage=asr_chunk_done job=abc123 chunk=0 duration_s=42.131 prepare_s=0.620 chunk_audio_s=1800.0 rtf=0.023 segments=180 parse_ok=1
[TIMING] stage=asr_chunk_done job=abc123 chunk=1 duration_s=45.221 prepare_s=0.640 chunk_audio_s=1800.0 rtf=0.025 segments=192 parse_ok=1
[TIMING] stage=asr_total job=abc123 duration_s=87.302 concurrency=12 num_chunks=2
[TIMING] stage=merge job=abc123 duration_s=0.054 segments=372 chars=12453
[TIMING] stage=job_done job=abc123 total_s=92.702 audio_s=3600.0 rtf=0.0258 gpu=NVIDIA L40S num_chunks=2 concurrency=12
```

切换 GPU 后重点对比 `stage=job_done` 的 `rtf`，以及响应里的
`timing.chunk_summary.transcription_slowest_chunk` 和
`timing.chunks[].transcription_time`。

## 6. 调参建议

- **真正决定吞吐的是 `transcription_slowest_chunk`**（并发下最慢那片），不是所有切片转写时间相加。
- 想降单切片延迟：提高业务 `concurrency`；`MAX_NUM_SEQS` 默认给足上限，通常不用改。
- 想降总成本：保持 `concurrency` ≤ Max Workers，避免排队。
- L40S 48 GB 起步：`MAX_MODEL_LEN=32768 MAX_NUM_SEQS=200 chunk_minutes=30 concurrency=12`。

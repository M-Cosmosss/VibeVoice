# VibeVoice ASR — RunPod Serverless 部署

端到端 ASR 服务：接收 `audio_url` → 切片 → 并发转写 → 合并 → 落盘 → 返回完整文稿。

## 1. 架构

- **基础镜像**：`vllm/vllm-openai:v0.14.1`，安装 VibeVoice plugin。
- **模型权重**：使用 RunPod Serverless **Model cache**，worker 从 `/runpod-volume/huggingface-cache/hub` 解析本地 snapshot，不把 14GB 权重烤进镜像。
- **服务端做端到端编排**：单次请求内完成音频下载、切片、并发 ASR、文稿合并，结果同时写入 `/tmp/transcripts/{job_id}.{json,txt}` 并在响应里返回（worker 销毁后丢失，但 API 已经把完整文稿返回了）。
- **日志**：每阶段输出 `[TIMING] stage=... duration_s=...` 结构化日志，便于切换 GPU benchmark。

### 镜像分层（按变更频率排序，最大化 cache 命中）

```
Layer 1: apt deps                         （几乎不变）
Layer 2: hf-transfer pip install          （几乎不变）
Layer 3: vibevoice / vllm_plugin 源码      （改动 vibevoice 时变）
Layer 4: pip install /app[vllm]           （依赖改动时变）
Layer 5: 生成 tokenizer patch 文件         （随 plugin 变，小层）
Layer 6: runpod/*.py 业务代码              （日常迭代只改这层）
```

日常改 `runpod/handler.py` / `runpod/pipeline.py` 只重建最后一层（~10 KB）。模型由 RunPod 的 Model cache 预热，不随镜像拉取。

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

首次构建 ≈ 15–25 min（多了 14 GB 模型下载层），增量构建几分钟。

## 3. RunPod Endpoint 配置

- **GPU**：L40S 48 GB（任意区域都可，因为模型烤在镜像里没区域绑定）
- **Active Workers**：`0`（scale-to-zero）
- **Max Workers**：按客户端最大并发上限设（默认建议 4）
- **Idle Timeout**：5–10s
- **FlashBoot**：ON
- **Container Disk**：20–30 GB（镜像不再包含 ASR 权重，仍需容纳 vLLM/CUDA 基础层）
- **Network Volume**：**不需要**
- **Model**：`microsoft/VibeVoice-ASR`（必须填写，用于启用 RunPod Model cache）
- **Execution Timeout**：根据最长音频估，参考下面的耗时表
- **Allow GPU fallback**：可选开（L40S 紧张时自动升档到 A100/H100，按使用计费）

### 环境变量（可选覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_MODEL_LEN` | `32768` | vLLM 上下文长度（30min 切片够用） |
| `MAX_NUM_SEQS` | `8` | vLLM 单 worker batch 上限 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM 显存占比 |
| `MODEL_ID` / `MODEL_NAME` | `microsoft/VibeVoice-ASR` | HF 模型 ID；RunPod Model 字段通常会注入 `MODEL_NAME` |
| `HF_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | RunPod cached model 根目录 |
| `MODEL_PATH` | `/tmp/vibevoice-asr-runtime` | 启动时创建的 runtime 模型目录，权重为 symlink，tokenizer 为镜像内 patch |
| `DEFAULT_CHUNK_MINUTES` | `30` | 切片长度（分钟） |
| `DEFAULT_CONCURRENCY` | `4` | 单请求内并发切片数 |
| `VLLM_READY_TIMEOUT_S` | `300` | 等待 vLLM 就绪超时 |
| `ASR_REQUEST_TIMEOUT_S` | `1800` | 单切片 ASR HTTP 超时 |
| `TRANSCRIPT_DIR` | `/tmp/transcripts` | 文稿落盘目录 |

## 4. 调用

### 4.1 同步 vs 异步

| 端点 | 何时用 | 上限 |
|---|---|---|
| `POST /v2/<EID>/runsync` | 估算总耗时 < ~280s 的请求（≤ 4h 音频在 L40S 上够用） | 默认 **300s** |
| `POST /v2/<EID>/run` + 轮询 `/status/{id}` | 长音频或不确定时长 | worker execution timeout，可配几小时 |

按 L40S + concurrency=4 估算：

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
    "concurrency": 4,
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
    "total_s": 92.7,
    "audio_duration_s": 3600.0,
    "rtf": 0.0258,
    "num_chunks": 2,
    "chunk_seconds": 1800,
    "concurrency": 4,
    "asr_per_chunk_s": [42.13, 45.22],
    "asr_total_s": 87.35,
    "asr_max_s": 45.22,
    "gpu": "NVIDIA L40S"
  }
}
```

> 注意：`files` 路径在 worker 容器内部，worker 销毁后丢失。如果要保留文稿历史请把 `text` / `segments` 入你自己的存储（S3/OSS/DB）。

### 4.4 同步调用示例

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://.../foo.mp3", "concurrency": 4}}'
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
[TIMING] stage=job_start job=abc123 chunk_minutes=30 concurrency=4 gpu=NVIDIA L40S
[TIMING] stage=download job=abc123 url=https://... duration_s=4.210 size_mb=58.30
[TIMING] stage=probe job=abc123 duration_s=0.082 audio_duration_s=3600.00
[TIMING] stage=split job=abc123 duration_s=1.103 num_chunks=2 chunk_seconds=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=0 start=0 end=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=1 start=1800 end=3600
[TIMING] stage=asr_chunk_done job=abc123 chunk=0 duration_s=42.131 chunk_audio_s=1800.0 rtf=0.023 segments=180 parse_ok=1
[TIMING] stage=asr_chunk_done job=abc123 chunk=1 duration_s=45.221 chunk_audio_s=1800.0 rtf=0.025 segments=192 parse_ok=1
[TIMING] stage=asr_total job=abc123 duration_s=87.302 concurrency=4 num_chunks=2
[TIMING] stage=merge job=abc123 duration_s=0.054 segments=372 chars=12453
[TIMING] stage=job_done job=abc123 total_s=92.702 audio_s=3600.0 rtf=0.0258 gpu=NVIDIA L40S num_chunks=2 concurrency=4
```

切换 GPU 后只需对比 `stage=job_done` 行的 `rtf`、`asr_max_s`、`asr_per_chunk_s` 即可。

## 6. 调参建议

- **真正决定吞吐的是 `asr_max_s`**（并发下最慢那片），不是 `asr_total_s`。
- 想降单切片延迟：增大 `MAX_NUM_SEQS`（更高显存占用） / 缩短 `chunk_minutes`（切更多片，并发分担）。
- 想降总成本：保持 `concurrency` ≤ Max Workers，避免排队。
- L40S 48 GB 起步：`MAX_MODEL_LEN=32768 MAX_NUM_SEQS=8 chunk_minutes=30 concurrency=4`。

# VibeVoice ASR — RunPod Serverless 部署

端到端 ASR 服务：接收 `audio_url` → 切片 → 并发转写 → 合并 → 落盘 → 返回完整文稿。

## 1. 架构

- **基础镜像**：`vllm/vllm-openai:v0.14.1`，安装 VibeVoice plugin。
- **模型权重**：放 RunPod Network Volume（`/runpod-volume/hf`），首次冷启动自动下载（~14 GB），后续冷启动直接读盘。
- **服务端做端到端编排**：单次请求内完成音频下载、切片、并发 ASR、文稿合并，结果同时写入 `/runpod-volume/transcripts/{job_id}.{json,txt}` 并在响应里返回。
- **日志**：每阶段输出 `[TIMING] stage=... duration_s=...` 结构化日志，便于切换 GPU benchmark。

## 2. 构建镜像

镜像必须从仓库根目录构建（Dockerfile 引用了 `vibevoice/`、`vllm_plugin/`、`pyproject.toml`、`runpod/`）：

```bash
cd /path/to/VibeVoice
docker build -f runpod/Dockerfile -t <your-registry>/vibevoice-asr-runpod:latest .
docker push <your-registry>/vibevoice-asr-runpod:latest
```

## 3. RunPod Endpoint 配置

### 3.1 创建 Network Volume

- Storage → Network Volumes → New Volume
- 大小：≥ 30 GB（模型 ~14 GB + 文稿空间）
- 区域：选你打算跑的 GPU 区域（必须同区域）

### 3.2 创建 Serverless Endpoint

- New Endpoint → Custom
- **Container Image**：`<your-registry>/vibevoice-asr-runpod:latest`
- **GPU**：L40S 48GB（推荐）
- **Active Workers**：`0`（真正 scale-to-zero）
- **Max Workers**：按客户端最大并发上限设（默认建议 4）
- **Idle Timeout**：5–10s
- **FlashBoot**：ON
- **Container Disk**：20 GB
- **Network Volume**：挂载到 `/runpod-volume`（路径必须一致）
- **Container Start Command**：留空（用 ENTRYPOINT）
- **环境变量**（可选覆盖默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_ID` | `microsoft/VibeVoice-ASR` | HF 模型 id |
| `MAX_MODEL_LEN` | `32768` | vLLM 上下文长度（30min 切片够用） |
| `MAX_NUM_SEQS` | `8` | vLLM 单 worker batch 上限 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM 显存占比 |
| `DEFAULT_CHUNK_MINUTES` | `30` | 切片长度（分钟） |
| `DEFAULT_CONCURRENCY` | `4` | 单请求内并发切片数 |
| `VLLM_READY_TIMEOUT_S` | `900` | 等待 vLLM 就绪超时 |
| `ASR_REQUEST_TIMEOUT_S` | `1800` | 单切片 ASR HTTP 超时 |

### 3.3 首次冷启动

- 第一个请求会触发权重下载（HF Hub → Network Volume），耗时约 5–15 分钟（取决于区域带宽）。
- 下载用 `flock` 单飞，多 worker 同时拉起也只会下载一次。
- 完成后写入 `/runpod-volume/hf/.vibevoice-prep.done`，后续 worker 直接跳过。
- 想避免首请求慢，可手动起一个临时 Pod 挂同一个 Network Volume，跑：

```bash
HF_HOME=/runpod-volume/hf python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/VibeVoice-ASR')"
python3 -m vllm_plugin.tools.generate_tokenizer_files --output $(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('microsoft/VibeVoice-ASR'))")
echo "$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('microsoft/VibeVoice-ASR'))")" > /runpod-volume/hf/.vibevoice-prep.done
```

## 4. 调用

### 4.1 输入

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

### 4.2 输出

```json
{
  "job_id": "abc123",
  "text": "完整合并后的文稿……",
  "segments": [
    {"start": 0.12, "end": 3.45, "speaker": "1", "text": "..."}
  ],
  "files": {
    "json": "/runpod-volume/transcripts/abc123.json",
    "txt":  "/runpod-volume/transcripts/abc123.txt"
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

### 4.3 调用示例（curl）

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://.../foo.mp3", "concurrency": 4}}'
```

> 长音频建议用 `/run`（异步）+ `/status/{id}` 轮询；`/runsync` 有 ~5 min 上限。

## 5. 日志（benchmark 友好）

每个阶段都会打 `[TIMING]` 行，可直接 grep 出来分析：

```
[TIMING] stage=vllm_ready duration_s=18.231
[TIMING] stage=job_start job=abc123 chunk_minutes=30 concurrency=4 gpu=NVIDIA L40S
[TIMING] stage=download job=abc123 url=https://... duration_s=4.210 size_mb=58.30
[TIMING] stage=probe job=abc123 duration_s=0.082 audio_duration_s=3600.00
[TIMING] stage=split job=abc123 duration_s=1.103 num_chunks=2 chunk_seconds=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=0 start=0 end=1800
[TIMING] stage=asr_chunk_start job=abc123 chunk=1 start=1800 end=3600
[TIMING] stage=asr_chunk_done job=abc123 chunk=0 duration_s=42.131 chunk_audio_s=1800.0 rtf=0.0234 segments=180 parse_ok=1
[TIMING] stage=asr_chunk_done job=abc123 chunk=1 duration_s=45.221 chunk_audio_s=1800.0 rtf=0.0251 segments=192 parse_ok=1
[TIMING] stage=asr_total job=abc123 duration_s=87.302 concurrency=4 num_chunks=2
[TIMING] stage=merge job=abc123 duration_s=0.054 segments=372 chars=12453
[TIMING] stage=job_done job=abc123 total_s=92.702 audio_s=3600.0 rtf=0.0258 gpu=NVIDIA L40S num_chunks=2 concurrency=4
```

切换 GPU 后只需对比 `stage=job_done` 行的 `rtf`、`asr_max_s`、`asr_per_chunk_s` 即可。

## 6. 调参建议

- **真正决定吞吐的是 `asr_max_s`**（并发下最慢那片），不是 `asr_total_s`。
- 想降单切片延迟：增大 `MAX_NUM_SEQS`（更高显存占用） / 缩短 `chunk_minutes`（切更多片，并发分担）。
- 想降总成本：保持 `concurrency` ≤ Max Workers，避免排队。
- L40S 48GB 建议起步：`MAX_MODEL_LEN=32768 MAX_NUM_SEQS=8 chunk_minutes=30 concurrency=4`。

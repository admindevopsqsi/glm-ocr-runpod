# GLM-OCR on RunPod Serverless

[![Runpod](https://api.runpod.io/badge/duozokker/glm-ocr-runpod)](https://console.runpod.io/hub/duozokker/glm-ocr-runpod)

Deploy [GLM-OCR](https://github.com/zai-org/GLM-OCR) (0.9B) as a serverless OCR endpoint on [RunPod](https://www.runpod.io). OpenAI-compatible API, auto-scaling from 0 to 100 workers.

## Features

- **Fast** — ~0.67 images/sec per worker with Multi-Token Prediction (MTP)
- **Cheap** — significantly cheaper than Google Document AI or AWS Textract
- **Scalable** — RunPod auto-scales 0→100 workers
- **Zero cold-start downloads** — model weights baked into the Docker image
- **OpenAI-compatible API** — works with the OpenAI SDK, cURL, or any HTTP client
- **GDPR-ready** — select EU data centers, no data sent to third-party AI providers

## Supported OCR Modes

| Prompt | Description |
|--------|-------------|
| `Text Recognition:` | Raw text extraction from images |
| `Formula Recognition:` | Mathematical formula recognition |
| `Table Recognition:` | Table structure extraction |
| `{"field": ""}` (JSON schema) | Structured information extraction |

## Deploy to RunPod

### Prerequisites

- A [RunPod account](https://www.runpod.io) with API key
- A GitHub account (repo can be public or private)

### Step 1: Connect GitHub to RunPod

1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings) → **Connections**
2. Click **Connect** on the GitHub card
3. Authorize RunPod to access your repositories

### Step 2: Fork this repo

Fork or clone this repo to your GitHub account. RunPod builds the Docker image directly from your repo.

### Step 3: Create the endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) → **New Endpoint**
2. Under **Import Git Repository**, select your forked repo
3. Select the `main` branch and leave the Dockerfile path as default
4. Click **Next**

### Step 4: Configure the endpoint

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| **Endpoint Name** | `glm-ocr` | Any name you like |
| **GPU** | RTX A4000 (16 GB) | Cheapest option that fits the 0.9B model |
| **Active Workers** | 0 | Set to 1 if you need zero cold starts |
| **Max Workers** | 5 | Adjust based on your volume |
| **Idle Timeout** | 5 seconds | How long a worker stays up after the last request |
| **FlashBoot** | Enabled | Free, reduces cold starts to ~500ms–2s |

5. Click **Deploy Endpoint**

### Step 5: Wait for the build

The first build takes **~15–20 minutes** (10.5 GB image). Track progress in the **Builds** tab. Subsequent builds are faster thanks to layer caching.

> **Tip:** To trigger a new build after pushing changes, create a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository). Standard pushes do not auto-trigger builds.

### Step 6: Get your endpoint ID

Once the build completes and a worker is running, copy your **Endpoint ID** from the endpoint details page. You'll need this for API calls.

## Usage

### Option A: Python test script

```bash
cp .env.example .env
# Edit .env with your RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID

pip install -r requirements.txt

# Text recognition
python test_endpoint.py --image https://example.com/document.png

# Table recognition
python test_endpoint.py --image ./invoice.jpg --prompt table

# Formula recognition
python test_endpoint.py --image ./equation.png --prompt formula

# Structured extraction with JSON schema
python test_endpoint.py --image ./id_card.png --prompt '{"name": "", "date_of_birth": "", "id_number": ""}'
```

### Option B: Batch processing

```bash
# Process all images in a directory (10 concurrent requests by default)
python batch_process.py ./documents/

# Table recognition with 20 concurrent requests
python batch_process.py ./documents/ --prompt table --concurrency 20

# Results saved to ./documents/ocr_output/
```

### Option C: cURL

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-OCR",
    "max_tokens": 16384,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/doc.png"}},
        {"type": "text", "text": "Text Recognition:"}
      ]
    }]
  }'
```

### Option D: OpenAI SDK (Python / JavaScript)

The endpoint is fully OpenAI-compatible. Use the standard OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1",
)

response = client.chat.completions.create(
    model="zai-org/GLM-OCR",
    max_tokens=16384,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/doc.png"}},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }],
)

print(response.choices[0].message.content)
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "YOUR_RUNPOD_API_KEY",
  baseURL: "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1",
});

const response = await client.chat.completions.create({
  model: "zai-org/GLM-OCR",
  max_tokens: 16384,
  messages: [{
    role: "user",
    content: [
      { type: "image_url", image_url: { url: "https://example.com/doc.png" } },
      { type: "text", text: "Text Recognition:" },
    ],
  }],
});

console.log(response.choices[0].message.content);
```

## Performance

Benchmarks from the [GLM-OCR paper](https://arxiv.org/html/2603.10910) (single GPU, single concurrency):

| Input Type | Speed | Per Page |
|------------|-------|----------|
| Images | 0.67 images/sec | ~1.5s |

> **Note:** The GLM-OCR paper also reports 1.86 PDF pages/sec for full document parsing
> with the official SDK pipeline (including layout detection via PP-DocLayoutV3). This repo
> serves only the base model — for full PDF pipeline performance, use the
> [official glmocr SDK](https://github.com/zai-org/GLM-OCR).

### Scaling

RunPod auto-scales workers based on incoming requests:

| Workers | Images/sec | Images/hour |
|---------|-----------|-------------|
| 1 | 0.67 | ~2,400 |
| 10 | 6.7 | ~24,000 |
| 50 | 33.5 | ~120,000 |
| 100 | 67 | ~240,000 |

## Cost

Using RunPod A4000 (16 GB) Flex workers at $0.00016/sec:

| Volume/month | Cost (compute only) |
|-------------|---------------------|
| 10,000 images | **~$2.40** |
| 100,000 images | **~$24** |
| 1,000,000 images | **~$240** |

Plus ~$1.05/month for container disk storage.

> **Real-world costs will be higher** than pure compute time. RunPod bills from worker
> start to stop, including cold start (~5–30s on first request) and idle timeout. For
> bursty or low-volume workloads, consider setting Active Workers to 1 to avoid cold
> starts, or batch your requests to maximize worker utilization.

For comparison: Google Document AI and AWS Textract charge ~$1.50 per 1,000 pages.

### GPU Options

| GPU | VRAM | Flex $/sec | Notes |
|-----|------|-----------|-------|
| **A4000** | 16 GB | $0.00016 | Best value for 0.9B model |
| **L4** | 24 GB | $0.00019 | More VRAM headroom |
| **A5000** | 24 GB | $0.00019 | Same tier as L4 |

## GDPR / DSGVO Compliance

RunPod is GDPR-compliant ([SOC 2 Type II](https://www.runpod.io/legal/compliance), [DPA](https://www.runpod.io/legal/data-processing-agreement)). To ensure compliance:

1. **Select EU data center** when creating your endpoint
2. The **DPA is automatically part of RunPod's ToS** — no separate signing needed
3. **No data leaves RunPod** — the model runs entirely within RunPod infrastructure
4. **No persistent storage** — vLLM processes images in-memory only
5. **HTTPS everywhere** — all API traffic is encrypted in transit

## Configuration

### Environment Variables

Set these in the RunPod endpoint settings to customize behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `zai-org/GLM-OCR` | HuggingFace model ID |
| `GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU VRAM to use |
| `MAX_MODEL_LEN` | auto | Maximum context length |
| `STARTUP_TIMEOUT` | `180` | Seconds to wait for vLLM to start |

### Customizing vLLM Args

For deeper customization, edit `handler.py` — the vLLM server command is built in `start_vllm()`. Push to GitHub and create a new release to trigger a rebuild.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  RunPod Worker Container                            │
│                                                     │
│  handler.py (RunPod SDK)                            │
│    │                                                │
│    ├── Starts vLLM server on localhost:8000          │
│    ├── Waits for /health to return 200              │
│    └── Forwards jobs to vLLM's OpenAI API           │
│                                                     │
│  vLLM Server (localhost:8000)                       │
│    └── Serves zai-org/GLM-OCR with MTP speculation  │
└─────────────────────────────────────────────────────┘
```

The Dockerfile:
1. Starts from `vllm/vllm-openai:nightly` (vLLM + CUDA runtime)
2. Installs `git`, `transformers` from main branch, and `runpod` SDK
3. Bakes model weights (~2 GB) into the image
4. Runs `handler.py` which starts vLLM and accepts RunPod jobs

> **Reproducibility:** The build uses floating tags (`nightly`, Git HEAD). For reproducible
> builds, pin `vllm/vllm-openai:<specific-tag>` and a transformers commit hash.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails at `git` | Ensure `apt-get install git` runs before pip install |
| `python` not found | Base image uses `python3`, not `python` |
| vLLM pip warning about transformers | Safe to ignore — works despite vLLM pinning `<5` |
| Cold start too slow | Enable FlashBoot + set Active Workers to 1 |
| Out of memory | Switch from A4000 (16 GB) to L4/A5000 (24 GB) |
| `.env` not loading | Run `pip install python-dotenv` or export vars directly |
| Build exceeds 160 min | RunPod build limit; try a faster base image tag |

## License

MIT — same as [GLM-OCR](https://github.com/zai-org/GLM-OCR).

# GLM-OCR Full Pipeline on RunPod

Deploy the official self-hosted [GLM-OCR](https://github.com/zai-org/GLM-OCR) pipeline on RunPod with:

- local `vLLM` for the GLM-OCR model
- the official `glmocr[selfhosted]` pipeline for PDFs, layout detection, tables, and formulas
- an HTTP service suitable for RunPod Load Balancing
- health and performance metrics
- benchmark tooling for large PDF sets

This repo is intentionally built around a plain HTTP service instead of a queue-based `runpod.serverless.start(...)` worker. The target deployment mode is RunPod Load Balancing or any always-on/custom HTTP endpoint where lower latency matters more than queue semantics.

## Why this version

The earlier queue-based design was fine for cheap image OCR, but it had three structural problems:

1. Cold starts were unavoidable with `Active Workers = 0`.
2. It only proxied the base OpenAI-compatible model route, not the official GLM-OCR PDF pipeline.
3. The service did not expose request-level throughput and cost metrics.

This repo fixes that by serving the official GLM-OCR self-hosted pipeline over HTTP and by preloading both required models into the image:

- `zai-org/GLM-OCR`
- `PaddlePaddle/PP-DocLayoutV3_safetensors`

## Endpoints

### `GET /health`

Returns startup phase and readiness. Suitable for load balancer health checks.

### `GET /metrics`

Returns aggregated request, page, timing, and estimated cost statistics.

### `POST /ocr/single`

Fast single-page OCR for:

- one image
- one remote image URL
- one rendered PDF page via `document + page`

This route is the explicit low-complexity fallback path and uses the stronger Markdown-preserving prompt inspired by `local-ocr-workbench`.

### `POST /glmocr/parse`

Runs the full GLM-OCR pipeline on a document.

Request body:

```json
{
  "document": "/path/to/file.pdf",
  "include_results": true,
  "save_layout_visualization": false
}
```

You can also pass `documents` as a list.

### `GET /openai/v1/models`
### `POST /openai/v1/chat/completions`

These routes remain available for raw `vLLM` access and debugging.

## Why it is simpler

The serving path is intentionally narrow and split into two modes:

1. start `vLLM`
2. wait for `/health`
3. start `glmocr[selfhosted]`
4. expose `/ocr/single` for fast single-page OCR
5. expose `/glmocr/parse` for full PDF/layout OCR

No second OCR provider, no MaaS dependency, no queue worker protocol, no browser-only PDF tricks.

## RunPod configuration

Use a RunPod Load Balancing endpoint, not a queue-based serverless worker.

Recommended starting point:

| Setting | Value |
|---|---|
| Endpoint Type | Load Balancing |
| Port | `8000` |
| Health Path | `/health` |
| GPU | A4000 16 GB or L4 |
| Active Workers | `0` or `1` depending on latency target |
| Idle Timeout | `60-180s` |
| Scaling Mode | Request count |
| FlashBoot | Enabled |

Tradeoff:

- `Active Workers = 0` is cheapest, but the first request after idle can still wait for boot.
- `Active Workers = 1` gives better latency but a materially higher monthly floor cost.

## Environment variables

These are the most important runtime settings:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `zai-org/GLM-OCR` | GLM-OCR model |
| `SERVED_MODEL_NAME` | `glm-ocr` | Model name exposed through `vLLM` |
| `GPU_MEMORY_UTILIZATION` | `0.95` | vLLM memory fraction |
| `MAX_MODEL_LEN` | `16384` | vLLM max model length |
| `MAX_NUM_SEQS` | unset | Optional vLLM concurrency cap |
| `SPECULATIVE_CONFIG` | `{"method":"mtp","num_speculative_tokens":1}` | Enables GLM MTP |
| `GLMOCR_LAYOUT_DEVICE` | `cpu` | Keep GPU free for OCR inference |
| `GPU_COST_PER_SEC` | `0.00016` | Used for cost estimation |
| `HF_TOKEN` | unset | Optional Hugging Face token for faster/more reliable downloads |

## Docker build

If you want Hugging Face-authenticated downloads during build:

```bash
docker build --build-arg HF_TOKEN="$HF_TOKEN" -t glmocr-runpod .
```

The image pre-downloads both GLM-OCR and the PP-DocLayout model to reduce boot time.

## Local run

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e GPU_COST_PER_SEC=0.00016 \
  glmocr-runpod
```

Wait until `/health` returns `200`.

## Benchmarking DATEV PDFs

This repo includes a benchmark script for the DATEV PDF folder:

```bash
python3 benchmark_datev.py \
  --base-url http://127.0.0.1:8000 \
  --input-dir /Users/schayan/Dev/MandantLink-v5/knowledge/books/datev-lehrbuecher \
  --output benchmark_results.json
```

The benchmark report includes:

- measured service time per document
- pages per second
- estimated cost per document
- estimated cost per 1000 documents
- estimated cost per 1000 pages

## Smoke tests

Smoke-test the running service with the receipt image from the official vLLM GLM-OCR recipe:

```bash
python3 smoke_test_service.py --base-url http://127.0.0.1:8000
```

And with a local PDF as well:

```bash
python3 smoke_test_service.py \
  --base-url http://127.0.0.1:8000 \
  --document /Users/schayan/Dev/MandantLink-v5/knowledge/books/datev-lehrbuecher/978-3-658-11659-0.pdf
```

## Files

- [service.py](/Users/schayan/Dev/GLM-5-OCR-Runpod/service.py)
- [glmocr.config.yaml](/Users/schayan/Dev/GLM-5-OCR-Runpod/glmocr.config.yaml)
- [benchmark_datev.py](/Users/schayan/Dev/GLM-5-OCR-Runpod/benchmark_datev.py)
- [Dockerfile](/Users/schayan/Dev/GLM-5-OCR-Runpod/Dockerfile)
- [CHANGELOG.md](/Users/schayan/Dev/GLM-5-OCR-Runpod/CHANGELOG.md)

## Notes

- `GLMOCR_LAYOUT_DEVICE=cpu` is the default because on a single-GPU machine it usually improves stability by reserving GPU memory for `vLLM`.
- If you care more about absolute throughput than stability, test `GLMOCR_LAYOUT_DEVICE=cuda`.
- The cost numbers in `/metrics` and `benchmark_datev.py` are estimates from measured processing time and your configured `GPU_COST_PER_SEC`. They do not include idle time or RunPod control-plane overhead.

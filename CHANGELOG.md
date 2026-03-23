# Changelog

## v0.3.0 - 2026-03-23

- Rebuilt the HTTP service around `FastAPI` and `uvicorn` instead of the Flask dev server
- Switched worker startup to a background thread so `/ping` can return `204` during initialization and `200` only when the full OCR stack is ready
- Kept `vLLM` and `glmocr[selfhosted]` startup separate, with cleaner shutdown and clearer error reporting
- Added a `/stats` alias for `/metrics` and a root route for quick inspection
- Made heavy runtime imports optional at module import time so local tests can run without the full OCR stack installed
- Added adaptive GPU defaults, minimum VRAM enforcement, and clearer RunPod Hub metadata for those settings
- Fixed client scripts to target the Load Balancer domain format `https://ENDPOINT_ID.api.runpod.ai`
- Removed the shared threaded `requests.Session` from batch processing to avoid client-side race issues
- Added explicit `fastapi`, `uvicorn`, `httpx`, and `pypdfium2` local dependencies and expanded service tests
- Enabled faster and more resilient Hugging Face downloads in the Docker image via env configuration

## v0.2.0 - 2026-03-23

- Replaced the queue-based worker with a simpler HTTP service in `service.py`
- Switched to the official `glmocr[selfhosted]` PDF/layout pipeline
- Added startup, throughput, and cost metrics
- Added DATEV benchmark tooling for real-world PDF measurements
- Preloaded both GLM-OCR and PP-DocLayoutV3 in the Docker image
- Updated the repo docs for RunPod Load Balancing deployment
- Strengthened the default OCR prompt for better fidelity on complex pages
- Cleaned the RunPod metadata and entrypoint shims

## v0.2.1 - 2026-03-23

- Added a dedicated `/ocr/single` endpoint for fast single-page OCR
- Kept `/glmocr/parse` as the explicit full document OCR path
- Imported the stronger Markdown-preserving fallback prompt from the local OCR workbench idea
- Added server-side PDF page rendering for single-page OCR fallback
- Added smoke tests using the public receipt example from the official vLLM GLM-OCR recipe

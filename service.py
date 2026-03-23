from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - depends on local runtime
    pdfium = None

try:
    from glmocr import GlmOcr
except ImportError:  # pragma: no cover - depends on local runtime
    GlmOcr = None

from prompts import SINGLE_OCR_PROMPT


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("glmocr-service")


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "80")))
VLLM_HOST = os.getenv("VLLM_HOST", "http://127.0.0.1:8080")
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "glm-ocr")
GPU_COST_PER_SEC = float(os.getenv("GPU_COST_PER_SEC", "0.00016"))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "900"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))
GLMOCR_CONFIG_PATH = os.getenv("GLMOCR_CONFIG_PATH", "/app/glmocr.config.yaml")
GLMOCR_LAYOUT_DEVICE = os.getenv("GLMOCR_LAYOUT_DEVICE", "cpu")
HEALTH_POLL_INTERVAL = float(os.getenv("HEALTH_POLL_INTERVAL", "1.0"))
SINGLE_OCR_PDF_DPI = int(os.getenv("SINGLE_OCR_PDF_DPI", "180"))
MIN_GPU_MEMORY_GB = float(os.getenv("MIN_GPU_MEMORY_GB", "16"))
UVICORN_LOG_LEVEL = os.getenv("UVICORN_LOG_LEVEL", "info")


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RequestSample:
    document: str
    elapsed_seconds: float
    pages: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.boot_started_at = time.time()
        self.vllm_ready_at: float | None = None
        self.pipeline_ready_at: float | None = None
        self.total_requests = 0
        self.total_documents = 0
        self.total_pages = 0
        self.total_request_seconds = 0.0
        self.total_cost_usd = 0.0
        self.last_samples: deque[RequestSample] = deque(maxlen=100)

    def mark_vllm_ready(self) -> None:
        with self.lock:
            self.vllm_ready_at = time.time()

    def mark_pipeline_ready(self) -> None:
        with self.lock:
            self.pipeline_ready_at = time.time()

    def record_request(self, documents: list[dict[str, Any]], elapsed_seconds: float) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_request_seconds += elapsed_seconds
            for document in documents:
                sample = RequestSample(
                    document=document["document"],
                    elapsed_seconds=float(document["elapsed_seconds"]),
                    pages=int(document["pages"]),
                    cost_usd=float(document["estimated_cost_usd"]),
                )
                self.total_documents += 1
                self.total_pages += sample.pages
                self.total_cost_usd += sample.cost_usd
                self.last_samples.append(sample)

    def is_vllm_ready(self) -> bool:
        with self.lock:
            return self.vllm_ready_at is not None

    def snapshot(self, stage: str, ready: bool) -> dict[str, Any]:
        with self.lock:
            avg_doc_seconds = self.total_request_seconds / self.total_documents if self.total_documents else None
            avg_page_seconds = self.total_request_seconds / self.total_pages if self.total_pages else None
            return {
                "stage": stage,
                "ready": ready,
                "uptime_seconds": round(time.time() - self.boot_started_at, 3),
                "boot": {
                    "started_at": self.boot_started_at,
                    "vllm_ready_after_seconds": None
                    if self.vllm_ready_at is None
                    else round(self.vllm_ready_at - self.boot_started_at, 3),
                    "pipeline_ready_after_seconds": None
                    if self.pipeline_ready_at is None
                    else round(self.pipeline_ready_at - self.boot_started_at, 3),
                },
                "totals": {
                    "requests": self.total_requests,
                    "documents": self.total_documents,
                    "pages": self.total_pages,
                    "request_seconds": round(self.total_request_seconds, 3),
                    "estimated_cost_usd": round(self.total_cost_usd, 6),
                    "avg_seconds_per_document": None if avg_doc_seconds is None else round(avg_doc_seconds, 3),
                    "avg_seconds_per_page": None if avg_page_seconds is None else round(avg_page_seconds, 3),
                    "estimated_cost_per_1000_documents_usd": None
                    if self.total_documents == 0
                    else round((self.total_cost_usd / self.total_documents) * 1000, 4),
                    "estimated_cost_per_1000_pages_usd": None
                    if self.total_pages == 0
                    else round((self.total_cost_usd / self.total_pages) * 1000, 4),
                },
                "recent_samples": [
                    {
                        "document": sample.document,
                        "elapsed_seconds": round(sample.elapsed_seconds, 3),
                        "pages": sample.pages,
                        "cost_usd": round(sample.cost_usd, 6),
                        "timestamp": sample.timestamp,
                    }
                    for sample in self.last_samples
                ],
            }


class ServiceState:
    def __init__(self) -> None:
        self.stage = "booting"
        self.ready = False
        self.vllm_process: subprocess.Popen[str] | None = None
        self.parser: Any | None = None
        self.metrics = Metrics()
        self.error: str | None = None
        self.runtime_profile: dict[str, Any] = {}
        self.startup_thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def set_stage(self, stage: str) -> None:
        with self.lock:
            self.stage = stage

    def set_error(self, error: str) -> None:
        with self.lock:
            self.error = error
            self.stage = "failed"
            self.ready = False

    def mark_ready(self) -> None:
        with self.lock:
            self.stage = "ready"
            self.ready = True
            self.error = None

    def set_runtime_profile(self, runtime_profile: dict[str, Any]) -> None:
        with self.lock:
            self.runtime_profile = runtime_profile

    def vllm_ready(self) -> bool:
        return self.metrics.is_vllm_ready()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = self.metrics.snapshot(self.stage, self.ready)
            if self.runtime_profile:
                data["runtime"] = self.runtime_profile
            if self.error:
                data["error"] = self.error
            return data


state = ServiceState()


def detect_gpu_info() -> tuple[str | None, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

    if not output:
        return None, None

    first_line = output.splitlines()[0]
    name, _, memory = first_line.partition(",")
    if not memory:
        return name.strip() or None, None
    try:
        return name.strip() or None, round(float(memory.strip()) / 1024, 2)
    except ValueError:
        return name.strip() or None, None


def resolve_runtime_profile() -> dict[str, Any]:
    gpu_name, gpu_memory_gb = detect_gpu_info()
    env_gpu_mem = os.getenv("GPU_MEMORY_UTILIZATION")
    env_max_len = os.getenv("MAX_MODEL_LEN")
    env_max_num_seqs = os.getenv("MAX_NUM_SEQS")
    env_enable_mtp = os.getenv("ENABLE_MTP")

    gpu_memory_utilization = env_gpu_mem
    max_model_len = env_max_len
    max_num_seqs = env_max_num_seqs
    enable_mtp = env_flag("ENABLE_MTP", gpu_memory_gb is None or gpu_memory_gb >= 24)
    notes: list[str] = []

    if gpu_memory_gb is not None and gpu_memory_gb < MIN_GPU_MEMORY_GB:
        raise RuntimeError(
            f"Detected GPU '{gpu_name or 'unknown'}' with {gpu_memory_gb:.2f} GB VRAM. "
            f"This image is configured for >= {MIN_GPU_MEMORY_GB:.0f} GB GPUs."
        )

    if gpu_memory_gb is not None:
        if gpu_memory_gb < 24:
            gpu_memory_utilization = gpu_memory_utilization or "0.88"
            max_model_len = max_model_len or "4096"
            max_num_seqs = max_num_seqs or "1"
            if env_enable_mtp is None:
                enable_mtp = False
            notes.append("Using the conservative 16 GB profile.")
        elif gpu_memory_gb < 32:
            gpu_memory_utilization = gpu_memory_utilization or "0.9"
            max_model_len = max_model_len or "8192"
            max_num_seqs = max_num_seqs or "2"
            notes.append("Using the balanced 24 GB profile.")
        else:
            gpu_memory_utilization = gpu_memory_utilization or "0.95"
            max_model_len = max_model_len or "16384"
            max_num_seqs = max_num_seqs or "2"
            notes.append("Using the high-memory 32 GB+ profile.")
    else:
        gpu_memory_utilization = gpu_memory_utilization or "0.9"
        max_model_len = max_model_len or "8192"
        notes.append("GPU VRAM could not be detected; using safe defaults.")

    return {
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "enable_mtp": enable_mtp,
        "layout_device": GLMOCR_LAYOUT_DEVICE,
        "min_gpu_memory_gb": MIN_GPU_MEMORY_GB,
        "notes": notes,
    }


def build_vllm_command(runtime_profile: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--port",
        "8080",
        "--gpu-memory-utilization",
        str(runtime_profile["gpu_memory_utilization"]),
        "--max-model-len",
        str(runtime_profile["max_model_len"]),
        "--allowed-local-media-path",
        "/",
    ]

    if env_flag("TRUST_REMOTE_CODE", True):
        cmd.append("--trust-remote-code")

    if runtime_profile["enable_mtp"]:
        cmd.extend(["--speculative-config.method", "mtp"])
        cmd.extend(["--speculative-config.num_speculative_tokens", os.getenv("NUM_SPECULATIVE_TOKENS", "1")])

    max_num_seqs = runtime_profile.get("max_num_seqs")
    if max_num_seqs:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])

    limit_mm_per_prompt = os.getenv("LIMIT_MM_PER_PROMPT")
    if limit_mm_per_prompt:
        cmd.extend(["--limit-mm-per-prompt", limit_mm_per_prompt])

    extra_args = os.getenv("VLLM_EXTRA_ARGS")
    if extra_args:
        cmd.extend(extra_args.split())

    return cmd


def wait_for_vllm() -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        process = state.vllm_process
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            response = requests.get(f"{VLLM_HOST}/health", timeout=2)
            if response.status_code == 200:
                state.metrics.mark_vllm_ready()
                return
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    raise TimeoutError(f"vLLM did not become healthy within {STARTUP_TIMEOUT}s")


def startup_worker() -> None:
    try:
        if GlmOcr is None:
            raise RuntimeError("Missing dependency 'glmocr'. Install glmocr[selfhosted,server] in the worker image.")

        runtime_profile = resolve_runtime_profile()
        state.set_runtime_profile(runtime_profile)

        state.set_stage("starting_vllm")
        cmd = build_vllm_command(runtime_profile)
        logger.info("Starting vLLM: %s", " ".join(cmd))
        state.vllm_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)
        wait_for_vllm()

        state.set_stage("starting_glmocr")
        state.parser = GlmOcr(
            config_path=GLMOCR_CONFIG_PATH,
            mode="selfhosted",
            layout_device=GLMOCR_LAYOUT_DEVICE,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        state.metrics.mark_pipeline_ready()
        state.mark_ready()
        logger.info("GLM-OCR service is ready")
    except Exception as exc:
        state.set_error(str(exc))
        logger.exception("Startup failed")


def ensure_startup_thread() -> None:
    with state.lock:
        thread = state.startup_thread
        if thread is not None and thread.is_alive():
            return
        state.startup_thread = threading.Thread(target=startup_worker, daemon=True, name="glmocr-startup")
        state.startup_thread.start()


def page_count_from_result(result: Any) -> int:
    json_result = getattr(result, "json_result", None)
    if isinstance(json_result, list):
        return len(json_result)
    return 1


def build_document_response(document: str, result: Any, elapsed_seconds: float) -> dict[str, Any]:
    pages = max(page_count_from_result(result), 1)
    cost = elapsed_seconds * GPU_COST_PER_SEC
    return {
        "document": document,
        "pages": pages,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "pages_per_second": round(pages / elapsed_seconds, 4) if elapsed_seconds > 0 else None,
        "estimated_cost_usd": round(cost, 6),
        "markdown_result": getattr(result, "markdown_result", ""),
        "json_result": getattr(result, "json_result", None),
        "original_images": getattr(result, "original_images", [document]),
    }


def save_result_if_requested(result: Any, output_dir: str | None, save_layout_visualization: bool) -> None:
    if not output_dir:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result.save(output_dir=output_dir, save_layout_visualization=save_layout_visualization)


def estimate_cost(elapsed_seconds: float) -> float:
    return elapsed_seconds * GPU_COST_PER_SEC


def local_path_from_input(value: str) -> Path:
    if value.startswith("file://"):
        return Path(value[7:])
    return Path(value)


def image_input_to_url(image: str) -> str:
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = local_path_from_input(image)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(suffix, "image/png")
    data = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def render_pdf_page_to_data_url(document: str, page: int, dpi: int = SINGLE_OCR_PDF_DPI) -> str:
    if pdfium is None:
        raise RuntimeError("Missing dependency 'pypdfium2'. Install it in the worker image.")

    path = local_path_from_input(document)
    pdf = pdfium.PdfDocument(str(path))
    try:
        if page < 1 or page > len(pdf):
            raise ValueError(f"Page {page} out of range for {document}")
        bitmap = pdf[page - 1].render(scale=dpi / 72).to_pil()
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            bitmap.save(tmp.name, format="PNG")
            return image_input_to_url(tmp.name)
    finally:
        pdf.close()


def build_single_ocr_payload(image_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def perform_single_ocr(image_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"{VLLM_HOST}/v1/chat/completions",
        json=build_single_ocr_payload(image_url, prompt, max_tokens),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    payload = response.json()
    content = ""
    choices = payload.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    return {
        "elapsed_seconds": round(elapsed, 3),
        "estimated_cost_usd": round(estimate_cost(elapsed), 6),
        "content": content,
        "raw_response": payload,
    }


def read_json_body(request: Request) -> dict[str, Any]:
    try:
        return request.state.json_payload  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return {}


def not_ready_response(require_parser: bool = True) -> JSONResponse:
    snapshot = state.snapshot()
    if require_parser:
        return JSONResponse(snapshot, status_code=503)
    if not state.vllm_ready():
        return JSONResponse(snapshot, status_code=503)
    return JSONResponse(snapshot, status_code=503)


def shutdown() -> None:
    parser = state.parser
    if parser is not None:
        try:
            parser.close()
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Failed to close GLM-OCR parser cleanly", exc_info=True)

    process = state.vllm_process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - best effort cleanup
            process.kill()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_startup_thread()
    yield
    shutdown()


app = FastAPI(title="GLM-OCR RunPod Service", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def capture_json_body(request: Request, call_next):
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            request.state.json_payload = await request.json()
        except Exception:
            request.state.json_payload = {}
    response = await call_next(request)
    return response


@app.get("/")
def root() -> dict[str, Any]:
    snapshot = state.snapshot()
    return {
        "service": "glmocr-runpod",
        "stage": snapshot["stage"],
        "ready": snapshot["ready"],
        "endpoints": {
            "ping": "/ping",
            "health": "/health",
            "metrics": "/metrics",
            "stats": "/stats",
            "single": "/ocr/single",
            "parse": "/glmocr/parse",
            "openai_models": "/openai/v1/models",
            "openai_chat_completions": "/openai/v1/chat/completions",
        },
    }


@app.get("/health")
@app.get("/ready")
def health() -> JSONResponse:
    snapshot = state.snapshot()
    if snapshot["ready"]:
        return JSONResponse(snapshot, status_code=200)
    return JSONResponse(snapshot, status_code=503)


@app.get("/ping")
def ping() -> Response:
    snapshot = state.snapshot()
    if snapshot["ready"]:
        return JSONResponse({"status": "healthy"}, status_code=200)
    if snapshot["stage"] == "failed":
        return JSONResponse({"status": "failed", "error": snapshot.get("error")}, status_code=503)
    return Response(status_code=204)


@app.get("/metrics")
@app.get("/stats")
def metrics() -> JSONResponse:
    return JSONResponse(state.snapshot(), status_code=200)


def proxy_openai_request(request: Request, route: str) -> Response:
    if not state.vllm_ready():
        return JSONResponse({"error": "vLLM is not ready yet."}, status_code=503)

    url = f"{VLLM_HOST}{route}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    body = getattr(request.state, "json_payload", None)
    data = None if body is not None else None
    if body is None:
        data = request._body if hasattr(request, "_body") else None  # pragma: no cover

    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            json=body,
            data=data,
            params=request.query_params,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    filtered_headers = {k: v for k, v in response.headers.items() if k.lower() not in excluded_headers}
    return Response(content=response.content, status_code=response.status_code, headers=filtered_headers)


@app.get("/openai/v1/models")
async def openai_models(request: Request) -> Response:
    return proxy_openai_request(request, "/v1/models")


@app.post("/openai/v1/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    return proxy_openai_request(request, "/v1/chat/completions")


@app.post("/ocr/single")
async def ocr_single(request: Request) -> Response:
    if not state.vllm_ready():
        return not_ready_response(require_parser=False)

    payload = read_json_body(request)
    prompt = payload.get("prompt") or SINGLE_OCR_PROMPT
    max_tokens = int(payload.get("max_tokens", 4096))
    page = int(payload.get("page", 1))
    image = payload.get("image")
    document = payload.get("document")

    if not image and not document:
        return JSONResponse({"error": "Expected 'image' or 'document' in request body."}, status_code=400)

    try:
        if image:
            image_url = image_input_to_url(image)
            source = image
        else:
            image_url = render_pdf_page_to_data_url(document, page)
            source = f"{document}#page={page}"

        result = perform_single_ocr(image_url, prompt, max_tokens)
        result["source"] = source
        result["mode"] = "single"
        result["prompt"] = prompt
        return JSONResponse(result, status_code=200)
    except Exception as exc:
        logger.exception("Single OCR failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/glmocr/parse")
async def glmocr_parse(request: Request) -> Response:
    if not state.ready or state.parser is None:
        return not_ready_response()

    payload = read_json_body(request)
    documents = payload.get("documents") or payload.get("images")
    if not documents:
        single = payload.get("document") or payload.get("image")
        documents = [single] if single else []
    documents = [document for document in documents if document]

    if not documents:
        return JSONResponse({"error": "Expected 'document' or 'documents' in request body."}, status_code=400)

    save_layout_visualization = bool(payload.get("save_layout_visualization", False))
    output_dir = payload.get("output_dir")
    include_results = bool(payload.get("include_results", True))

    request_started = time.perf_counter()
    parsed_documents: list[dict[str, Any]] = []

    try:
        for document in documents:
            document_started = time.perf_counter()
            result = state.parser.parse(document, save_layout_visualization=save_layout_visualization)
            document_elapsed = time.perf_counter() - document_started
            if output_dir:
                save_result_if_requested(result, output_dir, save_layout_visualization)
            parsed = build_document_response(document, result, document_elapsed)
            if not include_results:
                parsed.pop("markdown_result", None)
                parsed.pop("json_result", None)
            parsed_documents.append(parsed)
    except Exception as exc:
        logger.exception("Document parse failed")
        return JSONResponse({"error": str(exc)}, status_code=500)

    request_elapsed = time.perf_counter() - request_started
    state.metrics.record_request(parsed_documents, request_elapsed)

    total_pages = sum(document["pages"] for document in parsed_documents)
    total_cost = sum(document["estimated_cost_usd"] for document in parsed_documents)
    return JSONResponse(
        {
            "documents": parsed_documents,
            "summary": {
                "mode": "document",
                "documents": len(parsed_documents),
                "pages": total_pages,
                "elapsed_seconds": round(request_elapsed, 3),
                "pages_per_second": round(total_pages / request_elapsed, 4) if request_elapsed > 0 else None,
                "estimated_cost_usd": round(total_cost, 6),
                "estimated_cost_per_1000_documents_usd": round((total_cost / len(parsed_documents)) * 1000, 4),
                "estimated_cost_per_1000_pages_usd": round((total_cost / total_pages) * 1000, 4)
                if total_pages > 0
                else None,
            },
        },
        status_code=200,
    )


@app.post("/warmup")
async def warmup() -> Response:
    if not state.ready or state.parser is None:
        return not_ready_response()

    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9YlR5X0AAAAASUVORK5CYII="
    )
    with tempfile.TemporaryDirectory(prefix="glmocr_warmup_") as output_dir:
        result = state.parser.parse(tiny_png, save_layout_visualization=False)
        save_result_if_requested(result, output_dir, False)
    return JSONResponse({"status": "ok"}, status_code=200)


def main() -> None:
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level=UVICORN_LOG_LEVEL)


if __name__ == "__main__":
    main()

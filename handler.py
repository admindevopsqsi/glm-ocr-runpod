"""
RunPod Serverless handler for GLM-OCR via vLLM.

Key design: runpod.serverless.start() is called IMMEDIATELY so the worker
begins pinging RunPod right away.  vLLM is started in a background thread
and the handler waits for it to become healthy before processing jobs.
"""

import os
import subprocess
import sys
import threading
import time

import requests as http_requests
import runpod

VLLM_HOST = "http://localhost:8000"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "4096")
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "300"))

# Shared state
vllm_ready = threading.Event()
vllm_error = None


def start_vllm_background():
    """Start vLLM in a background thread and signal when ready."""
    global vllm_error

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--max-model-len", MAX_MODEL_LEN,
        "--allowed-local-media-path", "/",
    ]

    print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    for i in range(STARTUP_TIMEOUT):
        if proc.poll() is not None:
            vllm_error = f"vLLM exited with code {proc.returncode}"
            print(f"ERROR: {vllm_error}", flush=True)
            return
        try:
            r = http_requests.get(f"{VLLM_HOST}/health", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready after {i + 1}s", flush=True)
                vllm_ready.set()
                return
        except http_requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    vllm_error = f"vLLM failed to start within {STARTUP_TIMEOUT}s"
    print(f"ERROR: {vllm_error}", flush=True)


def handler(job):
    """Process a single RunPod job. Waits for vLLM if still starting."""
    if not vllm_ready.is_set():
        print("Waiting for vLLM to become ready...", flush=True)
        vllm_ready.wait(timeout=STARTUP_TIMEOUT)

    if vllm_error:
        return {"error": vllm_error}
    if not vllm_ready.is_set():
        return {"error": "vLLM not ready after timeout"}

    job_input = job["input"]

    # Format 1: openai_route + openai_input (RunPod vLLM worker compat)
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        body = job_input.get("openai_input", {})
    # Format 2: Direct OpenAI body
    elif "messages" in job_input:
        route = "/v1/chat/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    # Format 3: Raw prompt
    elif "prompt" in job_input:
        route = "/v1/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    else:
        return {"error": "Invalid input: expected 'messages', 'prompt', or 'openai_route'"}

    try:
        resp = http_requests.post(f"{VLLM_HOST}{route}", json=body, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except http_requests.RequestException as exc:
        return {"error": str(exc)}


# Start vLLM in background thread
threading.Thread(target=start_vllm_background, daemon=True).start()

# Start RunPod handler IMMEDIATELY so worker begins pinging
runpod.serverless.start({"handler": handler})

"""
RunPod Serverless handler for GLM-OCR via vLLM.

Starts a local vLLM server and forwards requests from
RunPod's queue API to the OpenAI-compatible endpoint.

Supports two input formats:
  1. openai_route + openai_input (RunPod vLLM worker format)
  2. Direct OpenAI chat completion body (messages, model, etc.)
"""

import os
import subprocess
import sys
import time

import requests
import runpod

VLLM_HOST = "http://localhost:8000"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.95")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "")
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "180"))


def start_vllm():
    """Start vLLM server as a subprocess and wait until healthy."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}',
    ]

    if MAX_MODEL_LEN:
        cmd.extend(["--max-model-len", MAX_MODEL_LEN])

    print(f"Starting vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

    # Wait for /health to return 200
    for i in range(STARTUP_TIMEOUT):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}")
        try:
            r = requests.get(f"{VLLM_HOST}/health", timeout=2)
            if r.status_code == 200:
                print(f"vLLM ready after {i+1}s")
                return proc
        except requests.ConnectionError:
            pass
        time.sleep(1)

    proc.kill()
    raise RuntimeError(f"vLLM failed to start within {STARTUP_TIMEOUT}s")


def handler(job):
    """Process a single RunPod job."""
    job_input = job["input"]

    # Format 1: openai_route + openai_input (RunPod vLLM worker compat)
    if "openai_route" in job_input:
        route = job_input["openai_route"]
        body = job_input.get("openai_input", {})
    # Format 2: Direct OpenAI body (messages present)
    elif "messages" in job_input:
        route = "/v1/chat/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    # Format 3: Raw prompt (text completion)
    elif "prompt" in job_input:
        route = "/v1/completions"
        body = job_input
        if "model" not in body:
            body["model"] = MODEL_NAME
    else:
        return {"error": "Invalid input: expected 'messages', 'prompt', or 'openai_route'"}

    try:
        resp = requests.post(f"{VLLM_HOST}{route}", json=body, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


# Start vLLM before accepting jobs
vllm_process = start_vllm()

runpod.serverless.start({"handler": handler})

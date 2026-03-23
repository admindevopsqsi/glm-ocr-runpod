#!/usr/bin/env python3
"""
Test client for the GLM-OCR service.

Supports:
  - Single-page OCR via /ocr/single
  - Full PDF/document parsing via /glmocr/parse
  - Remote files (URL), local files, and data URLs

Usage:
  python test_endpoint.py --image https://example.com/document.png
  python test_endpoint.py --image ./invoice.jpg --prompt table
  python test_endpoint.py --document ./book.pdf --mode parse

Environment (.env or exported):
  GLMOCR_BASE_URL      - Direct base URL to the HTTP service
  RUNPOD_API_KEY       - Optional RunPod API key for Load Balancing endpoint access
  RUNPOD_ENDPOINT_ID   - Optional RunPod endpoint ID for Load Balancing endpoint access
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional if env vars are exported directly

try:
    import requests
except ImportError:
    print("Missing dependency: pip install -r requirements.txt")
    sys.exit(1)


PROMPT_SHORTCUTS = {"text": None}

def resolve_prompt(raw: str) -> str:
    return PROMPT_SHORTCUTS.get(raw.lower(), raw)


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def call_single_route(
    base_url: str,
    image: str,
    prompt: str | None,
    api_key: str | None,
    timeout: int = 120,
) -> dict:
    """Call the single OCR route."""
    payload = {"image": image, "max_tokens": 4096}
    if prompt:
        payload["prompt"] = prompt
    url = f"{base_url.rstrip('/')}/ocr/single"
    headers = request_headers(api_key)
    start = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()

    print(f"\nTotal:     {elapsed:.2f}s")
    print(f"{'='*60}")

    print(result.get("content", ""))

    return result


def call_parse_route(base_url: str, document: str, api_key: str | None, timeout: int = 600) -> dict:
    url = f"{base_url.rstrip('/')}/glmocr/parse"
    payload = {"document": document, "include_results": True}
    print(f"Sending parse request to {url}...")
    start = time.time()
    resp = requests.post(url, json=payload, headers=request_headers(api_key), timeout=timeout)
    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    print(f"\nTotal:     {elapsed:.2f}s")
    print(f"{'='*60}")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Test the GLM-OCR HTTP service")
    parser.add_argument(
        "--image", help="Image path (local file or URL)"
    )
    parser.add_argument("--document", help="Document path or URL for full parse mode")
    parser.add_argument(
        "--prompt",
        default="text",
        help='Prompt: "text", "formula", "table", or custom string/JSON schema',
    )
    parser.add_argument(
        "--mode",
        choices=["single", "parse"],
        default="single",
        help="single=single page OCR fallback, parse=full GLM-OCR PDF pipeline",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("GLMOCR_BASE_URL"),
        help="Direct GLM-OCR base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RUNPOD_API_KEY"),
        help="RunPod API key for authenticated Load Balancer access",
    )
    parser.add_argument(
        "--endpoint-id",
        default=os.getenv("RUNPOD_ENDPOINT_ID"),
        help="RunPod Load Balancer endpoint ID",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds"
    )
    args = parser.parse_args()

    base_url = args.base_url
    if not base_url and args.api_key and args.endpoint_id:
        base_url = f"https://{args.endpoint_id}.api.runpod.ai"
    if not base_url:
        print("Error: provide --base-url or RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID")
        sys.exit(1)

    if args.mode == "parse":
        if not args.document:
            print("Error: --document required for --mode parse")
            sys.exit(1)
        print(f"Document: {args.document}")
        call_parse_route(base_url, args.document, args.api_key, args.timeout)
        return

    if not args.image:
        print("Error: --image required for --mode single")
        sys.exit(1)
    prompt = resolve_prompt(args.prompt)
    print(f"Image:  {args.image}")
    if prompt:
        print(f"Prompt: {prompt}")
    print()

    call_single_route(base_url, args.image, prompt, args.api_key, args.timeout)


if __name__ == "__main__":
    main()

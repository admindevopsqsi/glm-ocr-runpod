#!/usr/bin/env python3
"""
Batch-process images through the GLM-OCR HTTP service.

Uses the single-page OCR route:
  POST /ocr/single

Usage:
  python batch_process.py ./documents/
  python batch_process.py ./docs/ --prompt table
  python batch_process.py ./docs/ --concurrency 20

Environment (.env or exported):
  GLMOCR_BASE_URL     - Direct base URL to the service
  RUNPOD_API_KEY      - Optional RunPod API key
  RUNPOD_ENDPOINT_ID  - Optional RunPod Load Balancer endpoint ID
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("Missing dependency: pip install -r requirements.txt")
    sys.exit(1)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

PROMPT_SHORTCUTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}


def image_to_base64_url(path: Path) -> str:
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
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def process_image(
    base_url: str,
    image_path: Path,
    prompt: str,
    api_key: str | None,
    timeout: int = 300,
) -> dict:
    """Send a single image to RunPod and return the result."""
    image_url = image_to_base64_url(image_path)

    payload = {"image": image_url, "max_tokens": 4096}
    if prompt and prompt != "text":
        payload["prompt"] = prompt

    url = f"{base_url.rstrip('/')}/ocr/single"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()

    text = result.get("content", "")
    if not text:
        text = json.dumps(result, indent=2, ensure_ascii=False)

    return {"file": image_path, "text": text, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Batch OCR with GLM-OCR on RunPod")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: input_dir/ocr_output)",
    )
    parser.add_argument(
        "--prompt", default="text", help='"text", "formula", "table", or custom'
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max parallel requests"
    )
    parser.add_argument("--base-url", default=os.getenv("GLMOCR_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    args = parser.parse_args()

    base_url = args.base_url
    if not base_url and args.api_key and args.endpoint_id:
        base_url = f"https://{args.endpoint_id}.api.runpod.ai"
    if not base_url:
        print("Error: GLMOCR_BASE_URL or RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID required")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir else input_dir / "ocr_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_SHORTCUTS.get(args.prompt.lower(), args.prompt)

    files = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"No image files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} images")
    print(f"Prompt: {prompt}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_dir}")
    print()

    start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_image,
                base_url,
                f,
                prompt,
                args.api_key,
            ): f
            for f in files
        }
        for future in as_completed(futures):
            f = futures[future]
            try:
                result = future.result()
                text = result["text"]
                # Use full filename (with extension) to avoid collisions
                # e.g. invoice.png -> invoice.png.txt
                out_file = output_dir / f"{f.name}.txt"
                out_file.write_text(text, encoding="utf-8")
                completed += 1
                print(f"  Done: {f.name} ({len(text)} chars)")
            except Exception as e:
                failed += 1
                print(f"  Failed: {f.name}: {e}")

    elapsed = time.time() - start
    total = completed + failed
    pages_per_sec = completed / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Completed: {completed}/{total}")
    print(f"Failed:    {failed}/{total}")
    print(f"Time:      {elapsed:.1f}s")
    print(f"Speed:     {pages_per_sec:.2f} pages/sec (across workers)")
    print(f"Output:    {output_dir}")


if __name__ == "__main__":
    main()

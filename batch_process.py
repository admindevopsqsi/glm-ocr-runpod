#!/usr/bin/env python3
"""
Batch-process images through GLM-OCR on RunPod Serverless.

Sends concurrent requests to RunPod's OpenAI-compatible proxy endpoint.
RunPod auto-scales workers to handle parallel requests.

Usage:
  python batch_process.py ./documents/
  python batch_process.py ./docs/ --prompt table
  python batch_process.py ./docs/ --concurrency 20

Environment (.env or exported):
  RUNPOD_API_KEY      - Your RunPod API key
  RUNPOD_ENDPOINT_ID  - Your serverless endpoint ID
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
    session: requests.Session,
    api_key: str,
    endpoint_id: str,
    image_path: Path,
    prompt: str,
    timeout: int = 300,
) -> dict:
    """Send a single image to RunPod and return the result."""
    image_url = image_to_base64_url(image_path)

    payload = {
        "model": "zai-org/GLM-OCR",
        "max_tokens": 16384,
        "temperature": 0.01,
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

    url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = session.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()

    choices = result.get("choices", [])
    if choices:
        text = choices[0].get("message", {}).get("content", "")
    else:
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
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID"))
    args = parser.parse_args()

    if not args.api_key or not args.endpoint_id:
        print("Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID required")
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

    session = requests.Session()
    start = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_image,
                session,
                args.api_key,
                args.endpoint_id,
                f,
                prompt,
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

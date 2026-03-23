#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests
from pypdf import PdfReader


DEFAULT_INPUT_DIR = "/Users/schayan/Dev/MandantLink-v5/knowledge/books/datev-lehrbuecher"


def count_pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def benchmark_document(base_url: str, path: Path, timeout: int, include_results: bool) -> dict:
    started = time.perf_counter()
    headers = request_headers()
    response = requests.post(
        f"{base_url.rstrip('/')}/glmocr/parse",
        headers=headers,
        json={
            "document": str(path),
            "include_results": include_results,
            "save_layout_visualization": False,
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    document = payload["documents"][0]
    summary = payload["summary"]
    return {
        "document": str(path),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "pdf_pages": count_pdf_pages(path),
        "measured_elapsed_seconds": round(elapsed, 3),
        "service_elapsed_seconds": document["elapsed_seconds"],
        "service_pages": document["pages"],
        "service_pages_per_second": document["pages_per_second"],
        "estimated_cost_usd": document["estimated_cost_usd"],
        "summary": summary,
    }


def wait_until_ready(base_url: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    last_payload = {}
    headers = request_headers()
    while time.time() < deadline:
        try:
            ping = requests.get(f"{base_url.rstrip('/')}/ping", headers=headers, timeout=10)
            if ping.status_code == 200:
                response = requests.get(f"{base_url.rstrip('/')}/health", headers=headers, timeout=10)
                last_payload = response.json()
                if response.status_code == 200 and last_payload.get("ready"):
                    return last_payload
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Service at {base_url} did not become ready within {timeout}s. Last payload: {last_payload}")


def request_headers() -> dict[str, str]:
    token = os.getenv("RUNPOD_API_KEY") or os.getenv("BEARER_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def build_summary(results: list[dict]) -> dict:
    docs = len(results)
    total_pages = sum(item["pdf_pages"] for item in results)
    total_seconds = sum(item["service_elapsed_seconds"] for item in results)
    total_cost = sum(item["estimated_cost_usd"] for item in results)
    return {
        "documents": docs,
        "pages": total_pages,
        "total_service_seconds": round(total_seconds, 3),
        "total_estimated_cost_usd": round(total_cost, 6),
        "avg_seconds_per_document": round(total_seconds / docs, 3) if docs else None,
        "avg_seconds_per_page": round(total_seconds / total_pages, 3) if total_pages else None,
        "avg_pages_per_second": round(total_pages / total_seconds, 4) if total_seconds else None,
        "median_pages_per_second": round(statistics.median(item["service_pages_per_second"] for item in results), 4)
        if results
        else None,
        "estimated_cost_per_1000_documents_usd": round((total_cost / docs) * 1000, 4) if docs else None,
        "estimated_cost_per_1000_pages_usd": round((total_cost / total_pages) * 1000, 4) if total_pages else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLM-OCR service with DATEV PDFs")
    parser.add_argument("--base-url", default=os.getenv("GLMOCR_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--include-results", action="store_true")
    args = parser.parse_args()

    ready_payload = wait_until_ready(args.base_url, args.timeout)
    print(json.dumps({"health": ready_payload}, indent=2))

    input_dir = Path(args.input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        raise SystemExit(f"No PDFs found in {input_dir}")

    results = []
    for pdf in pdfs:
        print(f"Benchmarking {pdf.name} ...", flush=True)
        result = benchmark_document(args.base_url, pdf, args.timeout, args.include_results)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    report = {
        "health": ready_payload,
        "results": results,
        "summary": build_summary(results),
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved benchmark report to {args.output}")


if __name__ == "__main__":
    main()

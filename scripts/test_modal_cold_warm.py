#!/usr/bin/env python3
"""Measure cold-start vs warm-start latency on deployed Modal endpoint.

Usage:
  python3 scripts/test_modal_cold_warm.py \
    --base-url "https://<endpoint>.modal.run" \
    --audio-file "audio-example/audio-27sec-english.ogg"

Environment fallback:
  - BASE_URL from env or .env
  - API_KEY from env or .env
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".mp4", ".webm"}


def load_dotenv_if_present(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_credentials(base_url_arg: str | None, api_key_arg: str | None) -> tuple[str, str]:
    base_url = (base_url_arg or os.environ.get("BASE_URL") or "").strip().rstrip("/")
    api_key = (api_key_arg or os.environ.get("API_KEY") or "").strip()

    if not base_url:
        raise ValueError("Missing BASE_URL. Pass --base-url or set BASE_URL in env/.env")
    if not api_key:
        raise ValueError("Missing API_KEY. Pass --api-key or set API_KEY in env/.env")

    return base_url, api_key


def choose_audio_file(repo_root: Path, audio_file_arg: str | None, audio_dir_arg: str) -> Path:
    if audio_file_arg:
        audio_file = (repo_root / audio_file_arg).resolve()
        if not audio_file.exists() or not audio_file.is_file():
            raise ValueError(f"Audio file not found: {audio_file}")
        return audio_file

    audio_dir = (repo_root / audio_dir_arg).resolve()
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ValueError(f"Audio dir not found: {audio_dir}")

    for path in sorted(audio_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            return path

    raise ValueError(f"No supported audio files found in {audio_dir}")


def post_once(
    base_url: str,
    api_key: str,
    file_path: Path,
    model: str,
    response_format: str,
    timeout: int,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required. Install with: pip install requests") from exc

    url = f"{base_url}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "response_format": response_format}

    started = time.perf_counter()
    with file_path.open("rb") as fh:
        response = requests.post(
            url,
            headers=headers,
            data=data,
            files={"file": (file_path.name, fh, "application/octet-stream")},
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started

    text_length = 0
    try:
        payload = response.json()
        if isinstance(payload, dict):
            text_length = len(str(payload.get("text", "")))
    except Exception:
        text_length = len(response.text)

    return {
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "text_length": text_length,
        "ok": response.status_code == 200,
    }


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cold-vs-warm latency test for Modal-deployed Parakeet")
    parser.add_argument("--base-url", default=None, help="Modal endpoint base URL")
    parser.add_argument("--api-key", default=None, help="API key (Bearer token)")
    parser.add_argument("--audio-file", default=None, help="Single audio file path")
    parser.add_argument("--audio-dir", default="audio-example", help="Fallback audio directory")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3", help="Model form value")
    parser.add_argument("--response-format", default="json", choices=["json", "text"], help="Use text or json")
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout seconds")
    parser.add_argument("--cycles", type=int, default=3, help="Number of cold/warm cycles")
    parser.add_argument("--idle-seconds", type=int, default=25, help="Sleep before each cold request")
    parser.add_argument("--warm-runs", type=int, default=1, help="Immediate warm requests after each cold request")
    parser.add_argument(
        "--output-dir",
        default="modal-test-results/latency",
        help="Directory to store JSON report",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(repo_root / ".env")

    try:
        base_url, api_key = resolve_credentials(args.base_url, args.api_key)
        audio_file = choose_audio_file(repo_root, args.audio_file, args.audio_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Endpoint: {base_url}")
    print(f"Audio: {audio_file}")
    print(f"Cycles: {args.cycles}, warm-runs-per-cycle: {args.warm_runs}, idle-seconds: {args.idle_seconds}")

    report: dict[str, Any] = {
        "base_url": base_url,
        "audio_file": str(audio_file),
        "cycles": args.cycles,
        "idle_seconds": args.idle_seconds,
        "warm_runs": args.warm_runs,
        "runs": [],
    }

    cold_latencies: list[float] = []
    warm_latencies: list[float] = []
    failures = 0

    for cycle in range(1, args.cycles + 1):
        print(f"\nCycle {cycle}/{args.cycles}: sleeping {args.idle_seconds}s before cold request...")
        time.sleep(args.idle_seconds)

        cold = post_once(
            base_url=base_url,
            api_key=api_key,
            file_path=audio_file,
            model=args.model,
            response_format=args.response_format,
            timeout=args.timeout,
        )
        cold["phase"] = "cold"
        cold["cycle"] = cycle
        report["runs"].append(cold)

        if cold["ok"]:
            cold_latencies.append(float(cold["elapsed_seconds"]))
        else:
            failures += 1

        print(
            f"cold -> status={cold['status_code']} elapsed={cold['elapsed_seconds']:.2f}s "
            f"text_len={cold['text_length']}"
        )

        for warm_idx in range(1, args.warm_runs + 1):
            warm = post_once(
                base_url=base_url,
                api_key=api_key,
                file_path=audio_file,
                model=args.model,
                response_format=args.response_format,
                timeout=args.timeout,
            )
            warm["phase"] = "warm"
            warm["cycle"] = cycle
            warm["warm_index"] = warm_idx
            report["runs"].append(warm)

            if warm["ok"]:
                warm_latencies.append(float(warm["elapsed_seconds"]))
            else:
                failures += 1

            print(
                f"warm#{warm_idx} -> status={warm['status_code']} elapsed={warm['elapsed_seconds']:.2f}s "
                f"text_len={warm['text_length']}"
            )

    summary = {
        "cold_count": len(cold_latencies),
        "warm_count": len(warm_latencies),
        "cold_mean_seconds": round(mean_or_nan(cold_latencies), 3) if cold_latencies else None,
        "cold_median_seconds": round(median_or_nan(cold_latencies), 3) if cold_latencies else None,
        "warm_mean_seconds": round(mean_or_nan(warm_latencies), 3) if warm_latencies else None,
        "warm_median_seconds": round(median_or_nan(warm_latencies), 3) if warm_latencies else None,
        "failures": failures,
    }

    if cold_latencies and warm_latencies and mean_or_nan(warm_latencies) > 0:
        summary["cold_to_warm_mean_ratio"] = round(
            mean_or_nan(cold_latencies) / mean_or_nan(warm_latencies), 3
        )

    report["summary"] = summary

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_root / f"cold_warm_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"Report saved: {out_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

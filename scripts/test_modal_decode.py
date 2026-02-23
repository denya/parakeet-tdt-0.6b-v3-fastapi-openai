#!/usr/bin/env python3
"""Batch transcription smoke test against deployed Modal endpoint.

Usage examples:
  python3 scripts/test_modal_decode.py \
    --base-url "https://<endpoint>.modal.run" \
    --audio-dir "audio-example"

Environment fallback:
  - BASE_URL from env or .env
  - API_KEY from env or .env
"""

from __future__ import annotations

import argparse
import json
import os
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


def collect_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ValueError(f"Audio dir not found: {audio_dir}")

    files = [
        p for p in sorted(audio_dir.iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No supported audio files found in {audio_dir}")
    return files


def post_transcription(
    base_url: str,
    api_key: str,
    file_path: Path,
    model: str,
    response_format: str,
    timeout: int,
) -> tuple[int, float, str, dict[str, Any] | None, str]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required. Install with: pip install requests") from exc

    url = f"{base_url}/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "response_format": response_format}

    started = time.perf_counter()
    with file_path.open("rb") as fh:
        resp = requests.post(
            url,
            headers=headers,
            data=data,
            files={"file": (file_path.name, fh, "application/octet-stream")},
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started

    text_out = ""
    json_out: dict[str, Any] | None = None
    body = resp.text

    if response_format in {"text", "srt", "vtt"}:
        text_out = body.strip()
    else:
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                json_out = parsed
                text_out = str(parsed.get("text", "")).strip()
            else:
                text_out = ""
        except Exception:
            text_out = ""

    return resp.status_code, elapsed, text_out, json_out, body


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode smoke test for Modal-deployed Parakeet endpoint")
    parser.add_argument("--base-url", default=None, help="Modal endpoint base URL")
    parser.add_argument("--api-key", default=None, help="API key (Bearer token)")
    parser.add_argument("--audio-dir", default="audio-example", help="Directory with audio files")
    parser.add_argument("--model", default="parakeet-tdt-0.6b-v3", help="Model form value")
    parser.add_argument(
        "--response-format",
        default="text",
        choices=["json", "text", "srt", "vtt", "verbose_json"],
        help="Response format",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-request timeout seconds")
    parser.add_argument(
        "--output-dir",
        default="modal-test-results/decode",
        help="Directory to store raw outputs and summary",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty transcription text as success",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv_if_present(repo_root / ".env")

    try:
        base_url, api_key = resolve_credentials(args.base_url, args.api_key)
        audio_files = collect_audio_files((repo_root / args.audio_dir).resolve())
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.output_dir / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing endpoint: {base_url}")
    print(f"Audio files: {len(audio_files)}")
    print(f"Response format: {args.response_format}")
    print(f"Results dir: {run_dir}")

    summary: dict[str, Any] = {
        "base_url": base_url,
        "response_format": args.response_format,
        "model": args.model,
        "run_dir": str(run_dir),
        "files": [],
        "success": 0,
        "failed": 0,
    }

    for idx, audio_file in enumerate(audio_files, start=1):
        status, elapsed, text_out, json_out, raw_body = post_transcription(
            base_url=base_url,
            api_key=api_key,
            file_path=audio_file,
            model=args.model,
            response_format=args.response_format,
            timeout=args.timeout,
        )

        ok = status == 200 and (args.allow_empty or bool(text_out))
        if ok:
            summary["success"] += 1
        else:
            summary["failed"] += 1

        file_result = {
            "file": str(audio_file),
            "status_code": status,
            "elapsed_seconds": round(elapsed, 3),
            "text_length": len(text_out),
            "ok": ok,
        }

        output_base = run_dir / audio_file.stem
        if json_out is not None:
            (output_base.with_suffix(".json")).write_text(
                json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            (output_base.with_suffix(".txt")).write_text(raw_body, encoding="utf-8")

        summary["files"].append(file_result)

        preview = text_out.replace("\n", " ")[:120]
        print(
            f"[{idx}/{len(audio_files)}] {audio_file.name} | status={status} | "
            f"{elapsed:.2f}s | text_len={len(text_out)} | ok={ok} | preview={preview!r}"
        )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-")
    print(f"Success: {summary['success']} | Failed: {summary['failed']}")
    print(f"Summary: {summary_path}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

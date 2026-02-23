"""Modal deployment entrypoint for Parakeet TDT v3 OpenAI-compatible REST API.

Deploy:
  modal deploy modal_app.py

Serve (dev):
  modal serve modal_app.py

Client integration notes:
  - Backend decoding runs in 5-minute windows (`CHUNK_MINUTE = 5.0`).
  - For long recordings, split client uploads into ~30-45 minute chunks with
    small overlap, then merge transcripts client-side.
  - Modal web endpoints may return `303 See Other` roughly every 60s for long
    requests; clients must follow redirects until the final `200` response.
"""

import datetime
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave
from typing import Any, Optional

import modal

APP_NAME = "parakeet-tdt-0-6b-v3-openai"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_API_MODEL = "parakeet-tdt-0.6b-v3"
# Modal currently serves one NeMo backend model; keep API aliases for OpenAI-compatible clients.
API_MODEL_ALIASES = {
    "parakeet-tdt-0.6b-v3": MODEL_NAME,
    "istupakov/parakeet-tdt-0.6b-v3-onnx": MODEL_NAME,
    "grikdotnet/parakeet-tdt-0.6b-fp16": MODEL_NAME,
    "whisper-1": MODEL_NAME,
}

CHUNK_MINUTE = 5.0
SILENCE_THRESHOLD = "-40dB"
SILENCE_MIN_DURATION = 0.5
SILENCE_SEARCH_WINDOW = 30.0
SILENCE_DETECT_TIMEOUT = 300
MIN_SPLIT_GAP = 5.0

GPU_TYPE = os.environ.get("PARAKEET_MODAL_GPU", "l4")

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/cache",
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface-hub==0.36.0",
        "nemo_toolkit[asr]==2.3.2",
        "cuda-python==12.8.0",
        "fastapi==0.115.12",
        "python-multipart==0.0.20",
        "numpy<2",
        "pydub==0.25.1",
    )
    .entrypoint([])
)


def _extract_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> str:
    auth_header = (authorization or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    if auth_header:
        return auth_header
    return (x_api_key or "").strip()


def _run_command(command: list[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def _get_audio_duration(file_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = _run_command(command)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _detect_silence_points(
    file_path: str,
    silence_thresh: str = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_MIN_DURATION,
    total_duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    if not os.path.exists(file_path):
        return []

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        file_path,
        "-af",
        f"silencedetect=noise={silence_thresh}:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]

    try:
        result = _run_command(command, timeout=SILENCE_DETECT_TIMEOUT)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return []

    silence_points: list[tuple[float, float]] = []
    silence_start: Optional[float] = None

    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            try:
                silence_start = float(line.split("silence_start:")[1].split()[0])
            except (ValueError, IndexError):
                silence_start = None
        elif "silence_end:" in line and silence_start is not None:
            try:
                silence_end = float(line.split("silence_end:")[1].split()[0])
                silence_points.append((silence_start, silence_end))
                silence_start = None
            except (ValueError, IndexError):
                silence_start = None

    if silence_start is not None and total_duration is not None:
        silence_points.append((silence_start, total_duration))

    return silence_points


def _find_optimal_split_points(
    total_duration: float,
    target_chunk_duration: float,
    silence_points: list[tuple[float, float]],
    search_window: float = SILENCE_SEARCH_WINDOW,
    min_gap: float = MIN_SPLIT_GAP,
) -> list[float]:
    if not silence_points or total_duration <= target_chunk_duration:
        return []

    split_points: list[float] = []
    prev = 0.0
    num_chunks = math.ceil(total_duration / target_chunk_duration)

    for i in range(1, num_chunks):
        target_time = i * target_chunk_duration
        search_start = max(0.0, target_time - search_window)
        search_end = min(total_duration, target_time + search_window)

        candidates = [
            (start, end)
            for (start, end) in silence_points
            if start <= search_end and end >= search_start
        ]

        chosen: Optional[float] = None
        if candidates:
            candidates_sorted = sorted(
                candidates,
                key=lambda r: abs(((r[0] + r[1]) / 2.0) - target_time),
            )
            for start, end in candidates_sorted:
                split_point = (start + end) / 2.0
                if split_point > prev + min_gap and split_point <= total_duration - min_gap:
                    chosen = split_point
                    break

        if chosen is None:
            chosen = max(prev + min_gap, min(target_time, total_duration - min_gap))
            if chosen > total_duration:
                continue

        split_points.append(chosen)
        prev = chosen

    return split_points


def _format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=max(0.0, seconds))
    rendered = str(delta)

    if "." in rendered:
        integer_part, fractional_part = rendered.split(".", maxsplit=1)
        fractional_part = fractional_part[:3]
    else:
        integer_part = rendered
        fractional_part = "000"

    if len(integer_part.split(":")) == 2:
        integer_part = "0:" + integer_part

    return f"{integer_part},{fractional_part}"


def _segments_to_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, segment in enumerate(segments):
        text = str(segment.get("segment", "")).strip()
        if not text:
            continue
        lines.append(str(i + 1))
        lines.append(
            f"{_format_srt_time(float(segment['start']))} --> {_format_srt_time(float(segment['end']))}"
        )
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        text = str(segment.get("segment", "")).strip()
        if not text:
            continue
        start = _format_srt_time(float(segment["start"]))
        end = _format_srt_time(float(segment["end"]))
        lines.append(f"{start.replace(',', '.')} --> {end.replace(',', '.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _clean_text(text: str) -> str:
    cleaned = text.replace("\u2581", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.replace(" '", "'")


def _read_pcm16_bytes(wav_path: str) -> tuple[bytes, float]:
    with wave.open(wav_path, "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Expected mono audio after preprocessing")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Expected 16-bit audio after preprocessing")
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        data = wav_file.readframes(frames)
    duration = frames / float(sample_rate)
    return data, duration


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/cache": model_cache},
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    timeout=1800,
)
@modal.concurrent(max_inputs=1, target_inputs=1)
class ParakeetWorker:
    @modal.enter()
    def load(self):
        import logging

        import nemo.collections.asr as nemo_asr

        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    @modal.method()
    def transcribe_pcm16(self, audio_bytes: bytes) -> dict[str, Any]:
        import numpy as np

        if not audio_bytes:
            return {"text": "", "words": []}

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if audio_data.size == 0:
            return {"text": "", "words": []}

        with NoStdStreams():
            try:
                output = self.model.transcribe([audio_data], timestamps=True)
            except TypeError:
                output = self.model.transcribe([audio_data])

        if not output:
            return {"text": "", "words": []}

        hypothesis = output[0]
        if isinstance(hypothesis, str):
            text = hypothesis
        else:
            text = getattr(hypothesis, "text", str(hypothesis))

        words: list[dict[str, Any]] = []
        timestamps = getattr(hypothesis, "timestamp", None)
        if isinstance(timestamps, dict):
            for entry in timestamps.get("word", []):
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                word_text = str(entry[0]).strip()
                try:
                    start_time = float(entry[1])
                    end_time = float(entry[2])
                except (TypeError, ValueError):
                    continue
                if not word_text:
                    continue
                words.append({"word": word_text, "start": start_time, "end": end_time})

        return {"text": _clean_text(text), "words": words}


@app.cls(
    image=image,
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    secrets=[modal.Secret.from_name("parakeet-api-key")],
    timeout=600,
)
class ApiService:
    @modal.asgi_app()
    def web(self):
        from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import JSONResponse, PlainTextResponse

        api_key = os.environ.get("API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Missing API_KEY. Create secret with: modal secret create parakeet-api-key API_KEY=..."
            )

        worker = ParakeetWorker()
        chunk_duration_seconds = CHUNK_MINUTE * 60

        status_lock = threading.Lock()
        progress_tracker: dict[str, dict[str, Any]] = {}

        def enforce_api_key(
            authorization: Optional[str] = Header(default=None),
            x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        ) -> None:
            provided = _extract_api_key(authorization, x_api_key)
            if provided != api_key:
                raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

        web_app = FastAPI(title="Parakeet Transcription API", version="1.0.0")

        @web_app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "models": [k for k in API_MODEL_ALIASES.keys() if k != "whisper-1"],
                "default_model": DEFAULT_API_MODEL,
                "deployment": "modal",
                "backend_model": MODEL_NAME,
                "gpu": GPU_TYPE,
            }

        @web_app.get("/status")
        async def status(_: None = Depends(enforce_api_key)) -> dict[str, Any]:
            with status_lock:
                for job_id, progress in progress_tracker.items():
                    if progress.get("status") == "processing":
                        return {"job_id": job_id, **progress}

                if progress_tracker:
                    last_job_id = next(reversed(progress_tracker))
                    return {"job_id": last_job_id, **progress_tracker[last_job_id]}

            return {"status": "idle"}

        @web_app.post("/v1/audio/transcriptions")
        async def transcribe_audio(
            _: None = Depends(enforce_api_key),
            file: UploadFile = File(...),
            model: str = Form(default=DEFAULT_API_MODEL),
            response_format: str = Form(default="json"),
        ):
            model_name = (model or DEFAULT_API_MODEL).strip().lower()
            if model_name not in API_MODEL_ALIASES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model '{model_name}'. Allowed: {sorted(API_MODEL_ALIASES)}",
                )

            allowed_formats = {"json", "text", "srt", "verbose_json", "vtt"}
            if response_format not in allowed_formats:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid response_format '{response_format}'. Allowed: {sorted(allowed_formats)}",
                )

            if not file.filename:
                raise HTTPException(status_code=400, detail="No file selected")

            unique_id = str(uuid.uuid4())
            temp_dir = tempfile.mkdtemp(prefix=f"parakeet_{unique_id}_")

            with status_lock:
                progress_tracker[unique_id] = {
                    "status": "processing",
                    "current_chunk": 0,
                    "total_chunks": 0,
                    "progress_percent": 0,
                    "partial_text": "",
                }

            original_path = os.path.join(temp_dir, f"{unique_id}_upload")
            wav_path = os.path.join(temp_dir, f"{unique_id}.wav")

            try:
                uploaded = await file.read()
                if not uploaded:
                    raise HTTPException(status_code=400, detail="No file content")

                with open(original_path, "wb") as out_file:
                    out_file.write(uploaded)

                convert_cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    original_path,
                    "-ac",
                    "1",
                    "-ar",
                    str(TARGET_SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                ]

                try:
                    _run_command(convert_cmd)
                except subprocess.CalledProcessError as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"File conversion failed: {exc.stderr.strip()}",
                    ) from exc

                total_duration = _get_audio_duration(wav_path)
                if total_duration <= 0:
                    raise HTTPException(status_code=400, detail="Cannot process audio with 0 duration")

                split_points: list[float] = []
                if total_duration > chunk_duration_seconds:
                    silence_points = _detect_silence_points(wav_path, total_duration=total_duration)
                    if silence_points:
                        split_points = _find_optimal_split_points(
                            total_duration,
                            chunk_duration_seconds,
                            silence_points,
                            search_window=SILENCE_SEARCH_WINDOW,
                            min_gap=MIN_SPLIT_GAP,
                        )

                if split_points:
                    chunk_boundaries = [0.0, *split_points, total_duration]
                else:
                    num_chunks = math.ceil(total_duration / chunk_duration_seconds)
                    chunk_boundaries = [
                        min(i * chunk_duration_seconds, total_duration)
                        for i in range(num_chunks + 1)
                    ]

                num_chunks = max(1, len(chunk_boundaries) - 1)
                with status_lock:
                    progress_tracker[unique_id]["total_chunks"] = num_chunks

                all_segments: list[dict[str, Any]] = []
                all_words: list[dict[str, Any]] = []
                cumulative_time_offset = 0.0

                for i in range(num_chunks):
                    start_time = chunk_boundaries[i]
                    duration = chunk_boundaries[i + 1] - chunk_boundaries[i]
                    chunk_path = wav_path

                    if num_chunks > 1:
                        chunk_path = os.path.join(temp_dir, f"{unique_id}_chunk_{i}.wav")
                        chunk_cmd = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-ss",
                            str(start_time),
                            "-t",
                            str(duration),
                            "-i",
                            wav_path,
                            "-ac",
                            "1",
                            "-ar",
                            str(TARGET_SAMPLE_RATE),
                            "-c:a",
                            "pcm_s16le",
                            chunk_path,
                        ]
                        _run_command(chunk_cmd)

                    pcm_bytes, _ = _read_pcm16_bytes(chunk_path)
                    result = await worker.transcribe_pcm16.remote.aio(pcm_bytes)

                    cleaned_text = _clean_text(str(result.get("text", "")))
                    words = result.get("words") or []

                    if cleaned_text:
                        if words:
                            adjusted_words: list[dict[str, Any]] = []
                            for word_data in words:
                                try:
                                    word_start = float(word_data["start"]) + cumulative_time_offset
                                    word_end = float(word_data["end"]) + cumulative_time_offset
                                except (KeyError, TypeError, ValueError):
                                    continue
                                word_text = str(word_data.get("word", "")).strip()
                                if not word_text:
                                    continue
                                adjusted_words.append(
                                    {"start": word_start, "end": word_end, "word": word_text}
                                )

                            if adjusted_words:
                                all_words.extend(adjusted_words)
                                seg_start = adjusted_words[0]["start"]
                                seg_end = adjusted_words[-1]["end"]
                            else:
                                seg_start = cumulative_time_offset
                                seg_end = cumulative_time_offset + max(duration, 0.1)
                        else:
                            seg_start = cumulative_time_offset
                            seg_end = cumulative_time_offset + max(duration, 0.1)

                        all_segments.append(
                            {
                                "start": seg_start,
                                "end": seg_end,
                                "segment": cleaned_text,
                            }
                        )

                    with status_lock:
                        progress_tracker[unique_id]["current_chunk"] = i + 1
                        progress_tracker[unique_id]["progress_percent"] = int(
                            ((i + 1) / num_chunks) * 100
                        )
                        if cleaned_text:
                            progress_tracker[unique_id]["partial_text"] += cleaned_text + " "

                    cumulative_time_offset += duration

                full_text = " ".join(seg["segment"] for seg in all_segments).strip()

                with status_lock:
                    progress_tracker[unique_id]["status"] = "complete"
                    progress_tracker[unique_id]["progress_percent"] = 100

                    if len(progress_tracker) > 100:
                        for stale_job_id in list(progress_tracker.keys())[: len(progress_tracker) - 100]:
                            del progress_tracker[stale_job_id]

                if response_format == "text":
                    return PlainTextResponse(full_text)

                if response_format == "srt":
                    return PlainTextResponse(_segments_to_srt(all_segments), media_type="text/plain")

                if response_format == "vtt":
                    return PlainTextResponse(_segments_to_vtt(all_segments), media_type="text/plain")

                if response_format == "verbose_json":
                    return JSONResponse(
                        {
                            "task": "transcribe",
                            "language": "english",
                            "duration": total_duration,
                            "text": full_text,
                            "segments": [
                                {
                                    "id": idx,
                                    "seek": 0,
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "text": seg["segment"],
                                    "tokens": [],
                                    "temperature": 0.0,
                                    "avg_logprob": 0.0,
                                    "compression_ratio": 0.0,
                                    "no_speech_prob": 0.0,
                                }
                                for idx, seg in enumerate(all_segments)
                            ],
                            "words": all_words,
                            "model": model_name,
                        }
                    )

                response = JSONResponse({"text": full_text})
                response.headers["X-Job-ID"] = unique_id
                response.headers["X-Model"] = model_name
                return response

            except HTTPException:
                with status_lock:
                    progress_tracker[unique_id]["status"] = "failed"
                raise
            except Exception as exc:
                with status_lock:
                    progress_tracker[unique_id]["status"] = "failed"
                raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return web_app


class NoStdStreams:
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        import sys

        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        import sys

        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


@app.local_entrypoint()
def main() -> None:
    print("Run one of:")
    print("  modal serve modal_app.py")
    print("  modal deploy modal_app.py")
    print("\nSet secret first:")
    print("  modal secret create parakeet-api-key API_KEY=your-strong-key")

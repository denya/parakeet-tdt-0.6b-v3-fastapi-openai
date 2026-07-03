import sys

sys.stdout = sys.stderr

import gc
import os, sys, json, math, platform, queue, re, threading, time
import shutil
import uuid
import subprocess
import datetime
import psutil
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
from werkzeug.utils import secure_filename

import flask
from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
from pathlib import Path

host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", "5092"))
threads = int(os.environ.get("WAITRESS_THREADS", "8"))
listen = os.environ.get("PARAKEET_LISTEN", "").strip()
open_browser = os.environ.get("PARAKEET_OPEN_BROWSER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

CHUNK_MINUTE = float(os.environ.get("PARAKEET_CHUNK_MINUTE", "1.5"))

# Intelligent chunking configuration
SILENCE_THRESHOLD = os.environ.get("PARAKEET_SILENCE_THRESHOLD", "-40dB")
SILENCE_MIN_DURATION = float(os.environ.get("PARAKEET_SILENCE_MIN_DURATION", "0.5"))
SILENCE_SEARCH_WINDOW = float(os.environ.get("PARAKEET_SILENCE_SEARCH_WINDOW", "30.0"))
SILENCE_DETECT_TIMEOUT = int(os.environ.get("PARAKEET_SILENCE_DETECT_TIMEOUT", "300"))
MIN_SPLIT_GAP = float(os.environ.get("PARAKEET_MIN_SPLIT_GAP", "5.0"))

ROOT_DIR = Path(os.getcwd()).as_posix()


def _default_model_cache_dir() -> Path:
    configured_cache = (
        os.environ.get("PARAKEET_MODEL_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
    )
    if configured_cache:
        return Path(configured_cache).expanduser()

    repo_models_dir = Path(ROOT_DIR) / "models"
    if repo_models_dir.exists() or not repo_models_dir.is_symlink():
        return repo_models_dir

    return Path.home() / ".cache" / "parakeet-fastapi-openai" / "models"


MODEL_CACHE_DIR = _default_model_cache_dir()
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR.as_posix()
os.environ["HF_HUB_CACHE"] = MODEL_CACHE_DIR.as_posix()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
if sys.platform == "win32":
    os.environ["PATH"] = ROOT_DIR + f";{ROOT_DIR}/ffmpeg;" + os.environ["PATH"]

API_KEY = os.environ.get("API_KEY", "").strip()
if not API_KEY:
    print("❌ Missing required environment variable: API_KEY")
    sys.exit(1)


DEFAULT_MODEL_NAME = "parakeet-tdt-0.6b-v3"
MLX_MODEL_ID = os.environ.get("PARAKEET_MLX_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
ONNX_MODEL_ID = os.environ.get("PARAKEET_ONNX_MODEL", "nemo-parakeet-tdt-0.6b-v3")
VALID_BACKENDS = {"auto", "mlx", "onnx"}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _select_backend() -> str:
    requested = os.environ.get("PARAKEET_BACKEND", "auto").strip().lower()
    if requested not in VALID_BACKENDS:
        raise RuntimeError(
            f"Invalid PARAKEET_BACKEND={requested!r}; expected one of {sorted(VALID_BACKENDS)}"
        )
    if requested == "auto":
        return "mlx" if _is_apple_silicon() else "onnx"
    return requested


ACTIVE_BACKEND = _select_backend()


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _optional_float_env(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def _optional_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _parse_byte_size(value: str) -> int:
    value = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?i?b?|bytes?)?", value)
    if not match:
        raise ValueError(f"Invalid byte size: {value!r}")

    number = float(match.group(1))
    unit = match.group(2) or "b"
    unit_multipliers = {
        "": 1,
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    return int(number * unit_multipliers[unit])


def _optional_byte_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return _parse_byte_size(value)


def _format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unset"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024


MLX_MEMORY_LIMIT = _optional_byte_env("PARAKEET_MLX_MEMORY_LIMIT")
MLX_CACHE_LIMIT = _optional_byte_env("PARAKEET_MLX_CACHE_LIMIT")
MLX_WIRED_LIMIT = _optional_byte_env("PARAKEET_MLX_WIRED_LIMIT")
MAX_RSS_BYTES = _optional_byte_env("PARAKEET_MAX_RSS")
RSS_WATCH_INTERVAL = _float_env("PARAKEET_RSS_WATCH_INTERVAL", 5.0)
MAX_ACTIVE_TRANSCRIPTIONS = max(
    1,
    int(
        os.environ.get(
            "PARAKEET_MAX_ACTIVE_TRANSCRIPTIONS",
            "1" if ACTIVE_BACKEND == "mlx" else str(max(1, threads)),
        )
    ),
)
MAX_UPLOAD_MB = _float_env("PARAKEET_MAX_UPLOAD_MB", 2000.0)


def _configure_mlx_memory(mx_module) -> None:
    limits = [
        ("PARAKEET_MLX_MEMORY_LIMIT", MLX_MEMORY_LIMIT, mx_module.set_memory_limit),
        ("PARAKEET_MLX_CACHE_LIMIT", MLX_CACHE_LIMIT, mx_module.set_cache_limit),
        ("PARAKEET_MLX_WIRED_LIMIT", MLX_WIRED_LIMIT, mx_module.set_wired_limit),
    ]
    for env_name, limit, setter in limits:
        if limit is None:
            continue
        previous = setter(limit)
        print(
            f"MLX {env_name}={_format_bytes(limit)} "
            f"(previous {_format_bytes(previous)})"
        )


def _start_memory_watchdog() -> None:
    if MAX_RSS_BYTES is None:
        return

    def watch_rss():
        process = psutil.Process(os.getpid())
        while True:
            time.sleep(RSS_WATCH_INTERVAL)
            rss = process.memory_info().rss
            if rss > MAX_RSS_BYTES:
                print(
                    "RSS watchdog exiting for launchd restart: "
                    f"rss={_format_bytes(rss)} limit={_format_bytes(MAX_RSS_BYTES)}"
                )
                os._exit(75)

    threading.Thread(target=watch_rss, daemon=True, name="rss-watchdog").start()
    print(f"RSS watchdog limit: {_format_bytes(MAX_RSS_BYTES)}")


# Model configurations for different precision variants.
MODEL_CONFIGS = {
    DEFAULT_MODEL_NAME: {
        "hf_id": ONNX_MODEL_ID,
        "quantization": "int8",
        "mlx_hf_id": MLX_MODEL_ID,
        "description": "Parakeet TDT 0.6B v3",
    },
}

# Model cache for lazy loading
model_cache = {}
model_cache_lock = threading.Lock()


@dataclass
class CompatRecognitionResult:
    text: str
    tokens: List[str]
    timestamps: List[float]
    segments: Optional[List[dict]] = None
    words: Optional[List[dict]] = None


class OnnxParakeetBackend:
    backend = "onnx"

    def __init__(self, config: dict):
        print("\nInitializing ONNX Runtime...")
        import onnx_asr
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        print(f"Available providers: {available_providers}")
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError("CPUExecutionProvider is not available in onnxruntime.")

        providers_to_try = ["CPUExecutionProvider"]
        print(f"Using providers: {providers_to_try}")
        print("\nLoading Parakeet TDT 0.6B V3 ONNX model with INT8 quantization (CPU-only)...")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", "4"))
        sess_options.inter_op_num_threads = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._model = onnx_asr.load_model(
            config["hf_id"],
            quantization=config["quantization"],
            providers=providers_to_try,
            sess_options=sess_options,
        ).with_timestamps()
        print("ONNX model loaded successfully with CPU optimization!")

    def recognize(self, audio_path: str) -> Any:
        return self._model.recognize(audio_path)


class MlxParakeetBackend:
    backend = "mlx"

    def __init__(self, config: dict):
        self._jobs = queue.Queue()
        self._ready = threading.Event()
        self._startup_error = None
        self._mx = None
        self._worker = threading.Thread(
            target=self._worker_main,
            args=(config,),
            daemon=True,
            name="parakeet-mlx-worker",
        )
        self._worker.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def recognize(self, audio_path: str) -> CompatRecognitionResult:
        done = threading.Event()
        holder = {}
        self._jobs.put((audio_path, done, holder))
        done.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    def _worker_main(self, config: dict):
        try:
            print("\nInitializing MLX Runtime...")
            try:
                import mlx.core as mx
                from mlx.core import bfloat16, float32
                from parakeet_mlx import Beam, DecodingConfig, Greedy, SentenceConfig, from_pretrained
            except ImportError as exc:
                raise RuntimeError(
                    "MLX backend requires parakeet-mlx. Install it with "
                    "`uv pip install -r requirements-mlx.txt`."
                ) from exc
            self._mx = mx
            _configure_mlx_memory(mx)

            dtype_name = os.environ.get("PARAKEET_MLX_DTYPE", "bf16").strip().lower()
            if dtype_name not in {"bf16", "fp32"}:
                raise RuntimeError("PARAKEET_MLX_DTYPE must be bf16 or fp32")

            self._dtype = float32 if dtype_name == "fp32" else bfloat16
            self._chunk_duration = _float_env("PARAKEET_MLX_CHUNK_DURATION", 0.0)
            self._overlap_duration = _float_env("PARAKEET_MLX_OVERLAP_DURATION", 15.0)

            decoding = os.environ.get("PARAKEET_MLX_DECODING", "greedy").strip().lower()
            if decoding == "beam":
                decoding_strategy = Beam(
                    beam_size=int(os.environ.get("PARAKEET_MLX_BEAM_SIZE", "5")),
                    length_penalty=_float_env("PARAKEET_MLX_LENGTH_PENALTY", 0.013),
                    patience=_float_env("PARAKEET_MLX_PATIENCE", 3.5),
                    duration_reward=_float_env("PARAKEET_MLX_DURATION_REWARD", 0.67),
                )
            elif decoding == "greedy":
                decoding_strategy = Greedy()
            else:
                raise RuntimeError("PARAKEET_MLX_DECODING must be greedy or beam")

            self._decoding_config = DecodingConfig(
                decoding=decoding_strategy,
                sentence=SentenceConfig(
                    max_words=_optional_int_env("PARAKEET_MLX_MAX_WORDS"),
                    silence_gap=_optional_float_env("PARAKEET_MLX_SILENCE_GAP"),
                    max_duration=_optional_float_env("PARAKEET_MLX_MAX_DURATION"),
                ),
            )

            model_id = config["mlx_hf_id"]
            print(f"Loading MLX model: {model_id} ({dtype_name})")
            self._model = from_pretrained(
                model_id,
                dtype=self._dtype,
                cache_dir=MODEL_CACHE_DIR,
            )

            if os.environ.get("PARAKEET_MLX_LOCAL_ATTENTION", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                context_size = int(os.environ.get("PARAKEET_MLX_LOCAL_ATTENTION_CTX", "256"))
                self._model.encoder.set_attention_model(
                    "rel_pos_local_attn",
                    (context_size, context_size),
                )

            print("MLX model loaded successfully on Apple Silicon!")
            self._ready.set()

            while True:
                audio_path, done, holder = self._jobs.get()
                try:
                    holder["result"] = self._recognize_in_worker(audio_path)
                except Exception as exc:
                    holder["error"] = exc
                finally:
                    self._clear_runtime_cache("after transcribe")
                    done.set()
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()

    def _clear_runtime_cache(self, context: str) -> None:
        if self._mx is None:
            return
        try:
            gc.collect()
            active = self._mx.get_active_memory()
            cache = self._mx.get_cache_memory()
            peak = self._mx.get_peak_memory()
            self._mx.clear_cache()
            cache_after = self._mx.get_cache_memory()
            self._mx.reset_peak_memory()
            print(
                f"MLX memory {context}: active={_format_bytes(active)} "
                f"cache={_format_bytes(cache)} peak={_format_bytes(peak)} "
                f"cache_after_clear={_format_bytes(cache_after)}"
            )
        except Exception as exc:
            print(f"MLX cache cleanup failed: {exc}")

    def _recognize_in_worker(self, audio_path: str) -> CompatRecognitionResult:
        kwargs = {
            "dtype": self._dtype,
            "chunk_duration": self._chunk_duration if self._chunk_duration > 0 else None,
            "overlap_duration": self._overlap_duration,
            "decoding_config": self._decoding_config,
        }
        result = self._model.transcribe(audio_path, **kwargs)

        tokens = []
        timestamps = []
        segments = []
        words = []

        for sentence in getattr(result, "sentences", []) or []:
            sentence_text = getattr(sentence, "text", "").strip()
            sentence_start = float(getattr(sentence, "start", 0.0) or 0.0)
            sentence_end = float(getattr(sentence, "end", sentence_start) or sentence_start)
            if sentence_text:
                segments.append(
                    {
                        "start": sentence_start,
                        "end": sentence_end,
                        "segment": sentence_text,
                    }
                )

            for token in getattr(sentence, "tokens", []) or []:
                token_text = getattr(token, "text", "")
                token_start = float(getattr(token, "start", sentence_start) or sentence_start)
                token_end = float(getattr(token, "end", token_start) or token_start)
                tokens.append(token_text)
                timestamps.append(token_start)
                cleaned_token = token_text.replace("\u2581", " ").strip()
                if cleaned_token:
                    words.append(
                        {
                            "start": token_start,
                            "end": token_end,
                            "word": cleaned_token,
                        }
                    )

        text = getattr(result, "text", "") or ""
        if text and not segments:
            start_time = timestamps[0] if timestamps else 0.0
            end_time = words[-1]["end"] if words else start_time + 0.1
            segments.append({"start": start_time, "end": end_time, "segment": text})

        return CompatRecognitionResult(
            text=text,
            tokens=tokens,
            timestamps=timestamps,
            segments=segments,
            words=words,
        )


def get_model(model_name):
    """
    Get or load a model by name with lazy loading and caching.

    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)

    Returns:
        Loaded ASR model instance
    """
    # Default to Parakeet if model not found.
    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}', falling back to default model")
        model_name = DEFAULT_MODEL_NAME

    cache_key = (ACTIVE_BACKEND, model_name)
    with model_cache_lock:
        if cache_key in model_cache:
            print(f"Using cached {ACTIVE_BACKEND} model: {model_name}")
            return model_cache[cache_key]

        print(f"Loading {ACTIVE_BACKEND} model: {model_name}")
        config = MODEL_CONFIGS[model_name]
        if ACTIVE_BACKEND == "mlx":
            model = MlxParakeetBackend(config)
        elif ACTIVE_BACKEND == "onnx":
            model = OnnxParakeetBackend(config)
        else:
            raise RuntimeError(f"Unsupported backend: {ACTIVE_BACKEND}")

        model_cache[cache_key] = model
        return model


_start_memory_watchdog()

try:
    print(f"Selected Parakeet backend: {ACTIVE_BACKEND}")
    get_model(DEFAULT_MODEL_NAME)
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit()

print("=" * 50)


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "temp_uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)

# Progress tracking
progress_tracker = {}
progress_tracker_lock = threading.Lock()
PROGRESS_TTL_SECONDS = int(os.environ.get("PARAKEET_PROGRESS_TTL_SECONDS", "3600"))
PROGRESS_MAX_JOBS = int(os.environ.get("PARAKEET_PROGRESS_MAX_JOBS", "100"))
PROGRESS_PARTIAL_MAX_CHARS = int(
    os.environ.get("PARAKEET_PROGRESS_PARTIAL_MAX_CHARS", "20000")
)
transcription_slots = threading.BoundedSemaphore(MAX_ACTIVE_TRANSCRIPTIONS)
PROTECTED_PATHS = {"/status", "/metrics"}
PROTECTED_PREFIXES = ("/v1/", "/progress/")


def _trim_progress_text(text: str) -> str:
    if PROGRESS_PARTIAL_MAX_CHARS <= 0 or len(text) <= PROGRESS_PARTIAL_MAX_CHARS:
        return text
    return text[-PROGRESS_PARTIAL_MAX_CHARS:]


def _prune_progress_locked(now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    if PROGRESS_TTL_SECONDS > 0:
        cutoff = now - PROGRESS_TTL_SECONDS
        stale_job_ids = [
            job_id
            for job_id, progress in progress_tracker.items()
            if progress.get("status") != "processing"
            and progress.get("updated_at", progress.get("created_at", 0)) < cutoff
        ]
        for job_id in stale_job_ids:
            progress_tracker.pop(job_id, None)

    if PROGRESS_MAX_JOBS > 0 and len(progress_tracker) > PROGRESS_MAX_JOBS:
        removable = sorted(
            (
                (progress.get("updated_at", progress.get("created_at", 0)), job_id)
                for job_id, progress in progress_tracker.items()
                if progress.get("status") != "processing"
            )
        )
        overflow = len(progress_tracker) - PROGRESS_MAX_JOBS
        for _, job_id in removable[:overflow]:
            progress_tracker.pop(job_id, None)


def _set_progress(job_id: str, progress: dict) -> None:
    now = time.time()
    progress = {
        **progress,
        "created_at": now,
        "updated_at": now,
    }
    progress["partial_text"] = _trim_progress_text(progress.get("partial_text", ""))
    with progress_tracker_lock:
        _prune_progress_locked(now)
        progress_tracker[job_id] = progress


def _update_progress(job_id: str, **updates) -> None:
    now = time.time()
    with progress_tracker_lock:
        progress = progress_tracker.setdefault(
            job_id,
            {
                "status": "processing",
                "current_chunk": 0,
                "total_chunks": 0,
                "progress_percent": 0,
                "partial_text": "",
                "created_at": now,
            },
        )
        progress.update(updates)
        progress["updated_at"] = now
        progress["partial_text"] = _trim_progress_text(progress.get("partial_text", ""))
        _prune_progress_locked(now)


def _append_progress_text(job_id: str, text: str) -> None:
    if not text:
        return
    now = time.time()
    with progress_tracker_lock:
        progress = progress_tracker.get(job_id)
        if progress is None:
            return
        progress["partial_text"] = _trim_progress_text(
            progress.get("partial_text", "") + text
        )
        progress["updated_at"] = now


def _get_progress_snapshot(job_id: str) -> Optional[dict]:
    with progress_tracker_lock:
        _prune_progress_locked()
        progress = progress_tracker.get(job_id)
        return dict(progress) if progress is not None else None


def _get_processing_progress() -> Optional[Tuple[str, dict]]:
    with progress_tracker_lock:
        _prune_progress_locked()
        for job_id, progress in progress_tracker.items():
            if progress.get("status") == "processing":
                return job_id, dict(progress)
    return None


def _extract_api_key() -> str:
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    if auth_header:
        return auth_header
    return request.headers.get("X-API-Key", "").strip()


@app.before_request
def enforce_api_key():
    if request.method == "OPTIONS":
        return None

    path = request.path
    if path in PROTECTED_PATHS or path.startswith(PROTECTED_PREFIXES):
        provided_key = _extract_api_key()
        if provided_key != API_KEY:
            return jsonify({"error": "Unauthorized: invalid or missing API key"}), 401

    return None


def get_audio_duration(file_path: str) -> float:
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
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Could not get duration of file '{file_path}': {e}")
        return 0.0


def detect_silence_points(file_path: str, silence_thresh: str = SILENCE_THRESHOLD, 
                          silence_duration: float = SILENCE_MIN_DURATION,
                          total_duration: Optional[float] = None) -> List[Tuple[float, float]]:
    """
    Detect silence points in audio file using ffmpeg's silencedetect filter.
    
    Args:
        file_path: Path to audio file
        silence_thresh: Silence threshold in dB (e.g., "-40dB")
        silence_duration: Minimum silence duration in seconds
        total_duration: Total duration of audio (used to close trailing silence)
        
    Returns:
        List of tuples (silence_start, silence_end) in seconds
    """
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"Error: Audio file '{file_path}' not found for silence detection")
        return []
    
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", file_path,
        "-af", f"silencedetect=noise={silence_thresh}:d={silence_duration}",
        "-f", "null",
        "-"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=SILENCE_DETECT_TIMEOUT)
        
        # Parse stderr output for silence intervals
        silence_points = []
        silence_start = None
        
        for line in result.stderr.splitlines():
            if 'silence_start:' in line:
                try:
                    silence_start = float(line.split('silence_start:')[1].split()[0])
                except (ValueError, IndexError):
                    silence_start = None
            elif 'silence_end:' in line and silence_start is not None:
                try:
                    silence_end = float(line.split('silence_end:')[1].split()[0])
                    silence_points.append((silence_start, silence_end))
                    silence_start = None
                except (ValueError, IndexError):
                    pass
        
        # Close trailing silence if audio ended during silence
        if silence_start is not None and total_duration is not None:
            silence_points.append((silence_start, total_duration))
        
        return silence_points
    except subprocess.TimeoutExpired:
        print(f"Timeout: Silence detection exceeded {SILENCE_DETECT_TIMEOUT}s timeout")
        return []
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"Error running FFmpeg for silence detection: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error detecting silence: {e}")
        return []


def find_optimal_split_points(total_duration: float, target_chunk_duration: float, 
                               silence_points: List[Tuple[float, float]], 
                               search_window: float = SILENCE_SEARCH_WINDOW,
                               min_gap: float = MIN_SPLIT_GAP) -> List[float]:
    """
    Find optimal split points based on silence detection.
    
    Args:
        total_duration: Total audio duration in seconds
        target_chunk_duration: Target chunk size in seconds
        silence_points: List of (start, end) tuples for silence periods
        search_window: Search window in seconds around target split point
        min_gap: Minimum gap between split points to prevent 0-length chunks
        
    Returns:
        List of split points in seconds
    """
    if not silence_points or total_duration <= target_chunk_duration:
        return []
    
    split_points = []
    prev = 0.0
    num_chunks = math.ceil(total_duration / target_chunk_duration)
    
    for i in range(1, num_chunks):
        target_time = i * target_chunk_duration
        search_start = max(0.0, target_time - search_window)
        search_end = min(total_duration, target_time + search_window)
        
        # Find silence points that overlap with the search window
        candidates = [
            (start, end) for (start, end) in silence_points
            if start <= search_end and end >= search_start
        ]
        
        chosen = None
        if candidates:
            # Sort candidates by distance from target time
            candidates_sorted = sorted(
                candidates,
                key=lambda silence_range: abs(((silence_range[0] + silence_range[1]) / 2.0) - target_time)
            )
            # Find first candidate that satisfies minimum gap constraint
            for start, end in candidates_sorted:
                split_point = (start + end) / 2.0
                if split_point > prev + min_gap and split_point <= total_duration - min_gap:
                    chosen = split_point
                    break
        
        if chosen is None:
            # Fallback: target time, but enforce monotonicity and bounds
            chosen = max(prev + min_gap, min(target_time, total_duration - min_gap))
            # Ensure chosen doesn't exceed total_duration
            if chosen > total_duration:
                chosen = None  # Skip this split point if not feasible
        
        split_points.append(chosen)
        prev = chosen
    
    # Filter out None values if any splits were skipped
    split_points = [sp for sp in split_points if sp is not None]
    
    return split_points


def format_srt_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    s = str(delta)
    if "." in s:
        parts = s.split(".")
        integer_part = parts[0]
        fractional_part = parts[1][:3]
    else:
        integer_part = s
        fractional_part = "000"

    if len(integer_part.split(":")) == 2:
        integer_part = "0:" + integer_part

    return f"{integer_part},{fractional_part}"


def segments_to_srt(segments: list) -> str:
    srt_content = []
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"])
        end_time = format_srt_time(segment["end"])
        text = segment["segment"].strip()

        if text:
            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")

    return "\n".join(srt_content)


def segments_to_vtt(segments: list) -> str:
    vtt_content = ["WEBVTT", ""]
    for i, segment in enumerate(segments):
        start_time = format_srt_time(segment["start"]).replace(",", ".")
        end_time = format_srt_time(segment["end"]).replace(",", ".")
        text = segment["segment"].strip()

        if text:
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
    return "\n".join(vtt_content)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/parakeet.png")
def serve_logo():
    return flask.send_file("parakeet.png", mimetype="image/png")


@app.route("/health")
def health():
    available_models = list(MODEL_CONFIGS.keys())
    return jsonify({
        "status": "healthy",
        "models": available_models,
        "default_model": DEFAULT_MODEL_NAME,
        "backend": ACTIVE_BACKEND,
        "mlx_model": MLX_MODEL_ID if ACTIVE_BACKEND == "mlx" else None,
        "onnx_model": ONNX_MODEL_ID if ACTIVE_BACKEND == "onnx" else None,
    })


@app.route("/docs")
def swagger_ui():
    """Serve Swagger UI"""
    return render_template("swagger.html")


@app.route("/openapi.json")
def openapi_spec():
    """Return OpenAPI Specification"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Parakeet Transcription API",
            "description": "High-performance Parakeet speech transcription API compatible with OpenAI.",
            "version": "1.0.0"
        },
        "servers": [{"url": request.host_url.rstrip("/")}],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API Key",
                    "description": "Use API_KEY as: Authorization: Bearer <API_KEY>"
                }
            }
        },
        "paths": {
            "/v1/audio/transcriptions": {
                "post": {
                    "summary": "Transcribe Audio",
                    "description": "Transcribes audio into the input language. Supports real-time streaming progress.",
                    "operationId": "transcribe_audio",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "The audio file object (not file name) to transcribe."
                                        },
                                        "model": {
                                            "type": "string",
                                            "default": DEFAULT_MODEL_NAME,
                                            "enum": [DEFAULT_MODEL_NAME],
                                            "description": f"Model to use: {DEFAULT_MODEL_NAME}"
                                        },
                                        "response_format": {
                                            "type": "string",
                                            "default": "json",
                                            "enum": ["json", "text", "srt", "verbose_json", "vtt"],
                                            "description": "The format of the transcript output."
                                        }
                                    },
                                    "required": ["file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful Response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"}
                                        }
                                    }
                                },
                                "text/plain": {
                                    "schema": {"type": "string"}
                                }
                            }
                        },
                        "401": {
                            "description": "Unauthorized: invalid or missing API key"
                        }
                    }
                }
            }
        }
    })


@app.route("/progress/<job_id>")
def get_progress(job_id):
    """Get transcription progress for a job"""
    progress = _get_progress_snapshot(job_id)
    if progress is not None:
        return jsonify(progress)
    return jsonify({"status": "not_found"}), 404


@app.route("/status")
def get_status():
    """Get status of the most recent active job"""
    processing = _get_processing_progress()
    if processing is not None:
        job_id, progress = processing
        return jsonify({"job_id": job_id, **progress})
    return jsonify({"status": "idle"})


@app.route("/metrics")
def get_metrics():
    """Get real-time CPU and RAM metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    process_rss = psutil.Process(os.getpid()).memory_info().rss
    return jsonify({
        "cpu_percent": cpu_percent,
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "process_rss_gb": round(process_rss / (1024**3), 2),
        "process_max_rss_gb": (
            round(MAX_RSS_BYTES / (1024**3), 2) if MAX_RSS_BYTES else None
        ),
    })


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # OpenAI compatible parameters
    model_name = request.form.get("model", DEFAULT_MODEL_NAME).lower()
    response_format = request.form.get("response_format", "json").lower()
    legacy_srt_words = model_name == "parakeet_srt_words"

    print(f"Request Model: {model_name} | Format: {response_format}")

    if legacy_srt_words:
        model_name = DEFAULT_MODEL_NAME

    # Validate model and warn if unknown
    if model_name not in MODEL_CONFIGS:
        print(f"⚠️ Unknown model '{model_name}' requested, using default")
        model_name = DEFAULT_MODEL_NAME

    slot_acquired = transcription_slots.acquire(blocking=False)
    if not slot_acquired:
        return jsonify(
            {
                "error": "Too many active transcriptions",
                "details": (
                    "The local MLX backend is configured for "
                    f"{MAX_ACTIVE_TRANSCRIPTIONS} active transcription(s)."
                ),
            }
        ), 429

    original_filename = secure_filename(file.filename)

    unique_id = str(uuid.uuid4())
    temp_original_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{unique_id}_{original_filename}"
    )
    target_wav_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{unique_id}.wav")

    temp_files_to_clean = []

    try:
        # Get the appropriate model (with lazy loading)
        model_to_use = get_model(model_name)

        file.save(temp_original_path)
        temp_files_to_clean.append(temp_original_path)

        print(
            f"[{unique_id}] Converting '{original_filename}' to standard WAV format..."
        )
        ffmpeg_command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            temp_original_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            target_wav_path,
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return jsonify(
                {"error": "File conversion failed", "details": result.stderr}
            ), 500
        temp_files_to_clean.append(target_wav_path)

        CHUNK_DURATION_SECONDS = CHUNK_MINUTE * 60
        total_duration = get_audio_duration(target_wav_path)
        if total_duration == 0:
            return jsonify({"error": "Cannot process audio with 0 duration"}), 400

        # Use intelligent chunking based on silence detection
        chunk_paths = []
        split_points = []
        
        if total_duration > CHUNK_DURATION_SECONDS:
            print(f"[{unique_id}] Detecting silence points for intelligent chunking...")
            silence_points = detect_silence_points(target_wav_path, total_duration=total_duration)
            
            if silence_points:
                print(f"[{unique_id}] Found {len(silence_points)} silence periods")
                split_points = find_optimal_split_points(
                    total_duration, 
                    CHUNK_DURATION_SECONDS, 
                    silence_points,
                    search_window=SILENCE_SEARCH_WINDOW
                )
                print(f"[{unique_id}] Optimal split points: {[f'{sp:.2f}s' for sp in split_points]}")
            else:
                print(f"[{unique_id}] No silence detected, using time-based chunking")
        
        # Create chunks based on split points (or use time-based if no silence found)
        if split_points:
            # Silence-based chunking
            chunk_boundaries = [0.0] + split_points + [total_duration]
            num_chunks = len(chunk_boundaries) - 1
        else:
            # Time-based chunking (fallback)
            num_chunks = math.ceil(total_duration / CHUNK_DURATION_SECONDS)
            chunk_boundaries = [min(i * CHUNK_DURATION_SECONDS, total_duration) for i in range(num_chunks + 1)]
        
        # Initialize progress tracking
        _set_progress(
            unique_id,
            {
                "status": "processing",
                "current_chunk": 0,
                "total_chunks": num_chunks,
                "progress_percent": 0,
                "partial_text": "",
            },
        )
        
        print(
            f"[{unique_id}] Total duration: {total_duration:.2f}s. Splitting into {num_chunks} chunks."
        )

        if num_chunks > 1:
            for i in range(num_chunks):
                start_time = chunk_boundaries[i]
                duration = chunk_boundaries[i + 1] - start_time
                chunk_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"{unique_id}_chunk_{i}.wav"
                )
                chunk_paths.append(chunk_path)
                temp_files_to_clean.append(chunk_path)

                print(f"[{unique_id}] Creating chunk {i + 1}/{num_chunks} ({start_time:.2f}s - {chunk_boundaries[i+1]:.2f}s)...")
                chunk_command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-i",
                    target_wav_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    chunk_path,
                ]
                result = subprocess.run(chunk_command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Warning: Chunk extraction failed: {result.stderr}")
        else:
            chunk_paths.append(target_wav_path)

        all_segments = []
        all_words = []
        cumulative_time_offset = 0.0
        
        # Store chunk durations for offset calculation
        chunk_durations = []
        if num_chunks > 1:
            for i in range(num_chunks):
                duration = chunk_boundaries[i + 1] - chunk_boundaries[i]
                chunk_durations.append(duration)
        else:
            chunk_durations.append(total_duration)

        def clean_text(text):
            """Clean up spacing artifacts from token joining"""
            if not text:
                return ""
            # Handle potential SentencePiece underline
            text = text.replace("\u2581", " ")
            text = text.strip()
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            # Standard cleaning
            text = text.replace(" '", "'")
            return text

        def append_recognition_result(result, offset: float):
            if not result:
                return

            result_segments = getattr(result, "segments", None)
            result_words = getattr(result, "words", None)
            if result_segments:
                for result_segment in result_segments:
                    cleaned_segment = clean_text(result_segment.get("segment", ""))
                    if not cleaned_segment:
                        continue
                    segment = {
                        "start": float(result_segment.get("start", 0.0)) + offset,
                        "end": float(result_segment.get("end", 0.0)) + offset,
                        "segment": cleaned_segment,
                    }
                    all_segments.append(segment)
                    _append_progress_text(unique_id, cleaned_segment + " ")

                for result_word in result_words or []:
                    word_text = clean_text(result_word.get("word", ""))
                    if not word_text:
                        continue
                    all_words.append(
                        {
                            "start": float(result_word.get("start", 0.0)) + offset,
                            "end": float(result_word.get("end", 0.0)) + offset,
                            "word": word_text,
                        }
                    )
                return

            if not getattr(result, "text", ""):
                return

            start_time = result.timestamps[0] if result.timestamps else 0
            end_time = (
                result.timestamps[-1]
                if len(result.timestamps) > 1
                else start_time + 0.1
            )

            cleaned_text = clean_text(result.text)
            segment = {
                "start": start_time + offset,
                "end": end_time + offset,
                "segment": cleaned_text,
            }
            all_segments.append(segment)
            _append_progress_text(unique_id, cleaned_text + " ")

            for j, (token, timestamp) in enumerate(zip(result.tokens, result.timestamps)):
                if j < len(result.timestamps) - 1:
                    word_end = result.timestamps[j + 1]
                else:
                    word_end = end_time

                clean_token = token.replace("\u2581", " ").strip()
                word = {
                    "start": timestamp + offset,
                    "end": word_end + offset,
                    "word": clean_token,
                }
                all_words.append(word)

        for i, chunk_path in enumerate(chunk_paths):
            _update_progress(
                unique_id,
                current_chunk=i + 1,
                progress_percent=int((i + 1) / num_chunks * 100),
            )
            print(f"[{unique_id}] Transcribing chunk {i + 1}/{num_chunks}...")

            result = model_to_use.recognize(chunk_path)
            append_recognition_result(result, cumulative_time_offset)
            del result

            # Use planned chunk duration instead of ffprobe
            cumulative_time_offset += chunk_durations[i]

        print(f"[{unique_id}] All chunks transcribed, merging results.")
        
        # Update progress to complete
        _update_progress(unique_id, status="complete", progress_percent=100)

        if not all_segments:
            # Return empty structure if nothing found, consistent with failures or silence?
            # OpenAI sometimes returns empty json text.
            pass

        # Formatting Output
        full_text = " ".join([seg["segment"] for seg in all_segments])

        if response_format == "srt" or legacy_srt_words:
            srt_output = segments_to_srt(all_segments)
            if legacy_srt_words:
                json_str_list = [
                    {"start": it["start"], "end": it["end"], "word": it["word"]}
                    for it in all_words
                ]
                srt_output += "----..----" + json.dumps(json_str_list)
            return Response(srt_output, mimetype="text/plain")

        elif response_format == "vtt":
            return Response(segments_to_vtt(all_segments), mimetype="text/plain")

        elif response_format == "text":
            return Response(full_text, mimetype="text/plain")

        elif response_format == "verbose_json":
            # Minimal verbose_json structure
            return jsonify(
                {
                    "task": "transcribe",
                    "language": "english",  # detection not implemented here, hardcoded or param?
                    "duration": total_duration,
                    "text": full_text,
                    "segments": [
                        {
                            "id": idx,
                            "seek": 0,
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["segment"],
                            "tokens": [],  # Populate if needed
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 0.0,
                        }
                        for idx, seg in enumerate(all_segments)
                    ],
                }
            )

        else:
            # Default JSON
            response = jsonify({"text": full_text})
            response.headers['X-Job-ID'] = unique_id
            return response

    except Exception as e:
        print(f"A serious error occurred during processing: {e}")
        import traceback

        traceback.print_exc()
        _update_progress(unique_id, status="failed", error=str(e))
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        print(f"[{unique_id}] Cleaning up temporary files...")
        for f_path in temp_files_to_clean:
            if os.path.exists(f_path):
                os.remove(f_path)
        print(f"[{unique_id}] Temporary files cleaned.")
        transcription_slots.release()


def openweb():
    import webbrowser, time

    time.sleep(5)
    webbrowser.open_new_tab(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    listen_addresses = [
        item for item in re.split(r"[,\s]+", listen) if item
    ]
    print(f"Starting server...")
    print(f"Backend: {ACTIVE_BACKEND}")
    print(f"Web interface: http://127.0.0.1:{port}")
    if listen_addresses:
        print(f"Listening on: {', '.join(listen_addresses)}")
        api_host = listen_addresses[0]
    else:
        print(f"Listening on: {host}:{port}")
        api_host = f"{host}:{port}"
    print(f"API Endpoint: POST http://{api_host}/v1/audio/transcriptions")
    print(f"Running with {threads} threads.")
    if open_browser:
        print(f"Starting web browser thread...")
        threading.Thread(target=openweb).start()
    print(f"Starting waitress server...")
    print(f"Server ready.")
    if listen_addresses:
        serve(app, listen=listen_addresses, threads=threads)
    else:
        serve(app, host=host, port=port, threads=threads)

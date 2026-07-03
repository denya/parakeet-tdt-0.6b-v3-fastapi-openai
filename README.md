# Parakeet TDT Transcription with ONNX Runtime

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Parakeet TDT** is a high-performance implementation of NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model using [ONNX Runtime](https://onnxruntime.ai/), designed for ultra-fast inference on CPU.

This implementation achieves exceptional real-time speeds, outperforming standard [openai/whisper](https://github.com/openai/whisper) and competing directly with GPU-accelerated [faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementations while running entirely on consumer CPUs. The efficiency is achieved through the architectural advantages of the Token-and-Duration Transducer (TDT) model combined with 8-bit quantization.

## 🌍 Multilingual Support

**Parakeet TDT 0.6B v3** features robust multilingual capabilities with **automatic language detection**. The model can automatically identify and transcribe speech in any of the **25 supported languages** without requiring manual language specification:

English, Spanish, French, Russian, German, Italian, Polish, Ukrainian, Romanian, Dutch, Hungarian, Greek, Swedish, Czech, Bulgarian, Portuguese, Slovak, Croatian, Danish, Finnish, Lithuanian, Slovenian, Latvian, Estonian, Maltese

Simply send audio in any of these languages, and the model will automatically detect and transcribe it with high accuracy, including proper punctuation and capitalization.

## Benchmark

### LibriSpeech test-clean (Verified Ground Truth) ⭐

Benchmarked on **LibriSpeech test-clean** dataset with professionally verified human transcriptions. This provides reliable, reproducible accuracy metrics.

**Test Environment:** CPU-only inference, 50 samples (~350 seconds of audio)

| Model | Precision | Accuracy | WER | CER | Speedup (RTF) |
|-------|-----------|----------|-----|-----|---------------|
| **Parakeet TDT 0.6B v3** | INT8 | **97.84%** | 2.16% | 0.56% | **18.41x** (0.054) |
| **Parakeet TDT 0.6B v3** | FP16 | **97.84%** | 2.16% | 0.56% | **18.82x** (0.053) |
| **Parakeet TDT 0.6B v3** | FP32 | **97.84%** | 2.16% | 0.56% | **19.42x** (0.052) |
| Whisper Large v3* | FP16 | ~95-96% | ~4-5% | ~2-3% | varies |

> *Whisper Large v3 benchmarks from published literature on LibriSpeech test-clean. Actual results vary by implementation and hardware.

**Key Findings:**
- All Parakeet precision variants achieve **identical accuracy** (97.84%)
- INT8 quantization has **zero accuracy loss** vs FP32
- Real-time factor (RTF) of ~0.05 means 20x faster than real-time
- Competitive with Whisper Large v3 accuracy with significantly faster CPU inference

---

### Parakeet TDT vs Faster Whisper

We compare the performance of **Parakeet TDT (CPU)** against **faster-whisper (GPU & CPU)**.

The metric used is **Speedup Factor** (Audio Duration / Processing Time). Higher is better.

| Implementation | Hardware | Model | Precision | Speedup |
| --- | --- | --- | --- | --- |
| **Parakeet TDT** (Ours) | **CPU** (i7-12700KF) | **TDT 0.6B v3** | **int8** | **~29.7x** |
| **Parakeet TDT** (Ours) | **CPU** (i7-4790) | **TDT 0.6B v3** | **int8** | **~17.0x** |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | int8 | 13.2x |
| faster-whisper | GPU (RTX 3070 Ti) | Large-v2 | fp16 | 12.4x |
| faster-whisper | CPU (i7-12700K) | Small | int8 | 7.6x |
| faster-whisper | CPU (i7-12700K) | Small | fp32 | 4.9x |

*   **Parakeet TDT**: Benchmarked on Intel Core i7-12700K with ONNX Runtime INT8.
*   **faster-whisper**: Benchmarks from [official faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper).

### Detailed Parakeet Performance

| Metrics | Result |
| --- | --- |
| **Average Speedup** | **29.7x** |
| **Real Time Factor (RTF)** | **0.033** |
| **Max Speedup** | **~30x** |

### Extended Multilingual Benchmark (YouTube Samples)

Additional benchmark on real-world YouTube content across multiple languages:

| Language | Model Variant | Latency (s) | Speedup (RTF) | WER | CER |
| --- | --- | ---: | ---: | ---: | ---: |
| English | INT8 (`parakeet-tdt-0.6b-v3`) | 70.60 | 20.32x (0.049) | 5.13% | 2.35% |
| English | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 135.43 | 10.59x (0.094) | 5.48% | 2.83% |
| English | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 112.80 | 12.72x (0.079) | 5.53% | 2.85% |
| English | Whisper-Large-v3 (DeepInfra) | 53.45 | 26.84x (0.037) | 4.25% | 3.91% |
| Spanish | INT8 (`parakeet-tdt-0.6b-v3`) | 29.92 | 18.64x (0.054) | 19.45% | 13.79% |
| Spanish | FP16 (`grikdotnet/parakeet-tdt-0.6b-fp16`) | 48.52 | 11.49x (0.087) | 15.31% | 11.33% |
| Spanish | FP32 (`istupakov/parakeet-tdt-0.6b-v3-onnx`) | 38.99 | 14.30x (0.070) | 15.31% | 11.33% |
| Spanish | Whisper-Large-v3 (DeepInfra) | 15.79 | 35.30x (0.028) | 20.70% | 18.05% |

> ⚠️ **Note:** YouTube subtitle references may contain errors. For verified accuracy, see LibriSpeech benchmark above.

## Requirements

*   [Docker](https://docs.docker.com/get-docker/) (Recommended)
*   Or: Python 3.10+ and [FFmpeg](https://ffmpeg.org/)

### CPU Optimization
For hybrid CPUs (like Intel 12th-14th Gen), performance is significantly improved by pinning the process to Performance cores (P-cores).

## Installation

### 🐳 Docker (Recommended)

The easiest way to get started. No dependencies to install!

**CPU Deployment:**
```bash
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
export API_KEY="replace-with-strong-key"
docker compose up parakeet-cpu -d
```

**GPU Deployment** (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
export API_KEY="replace-with-strong-key"
docker compose up parakeet-gpu -d
```

The server will be available at `http://localhost:5092`. See [DOCKER.md](DOCKER.md) for more options.

---

### Conda (Alternative)

For development or customization:

```bash
conda create -n parakeet-onnx python=3.10
conda activate parakeet-onnx
git clone https://github.com/groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai
cd parakeet-tdt-0.6b-v3-fastapi-openai
pip install -r requirements.txt
```

### Apple Silicon MLX (macOS, no Docker)

For local Apple Silicon inference, use the MLX backend. It keeps the same
OpenAI-compatible API but loads `mlx-community/parakeet-tdt-0.6b-v3` through
`parakeet-mlx`.

```bash
cd ~/code/parakeet-tdt-0.6b-v3-fastapi-openai
export API_KEY="replace-with-strong-key"
./scripts/run_mlx_macos.sh
```

If `API_KEY` is not already exported, the launcher will also read it from the
repo `.env` file. The launcher creates `.venv-mlx`, installs
`requirements-mlx.txt`, uses `~/.cache/parakeet-fastapi-openai/models` for model
downloads, and binds to localhost plus your Tailscale IPv4 address when the
Tailscale CLI is available. The launcher checks `PATH` first, then the standard
macOS app CLI path at `/Applications/Tailscale.app/Contents/MacOS/Tailscale`.

Useful overrides:

```bash
export PORT=5092
export PARAKEET_BACKEND=mlx
export PARAKEET_LISTEN="127.0.0.1:5092 100.x.y.z:5092"
export PARAKEET_MLX_DTYPE=bf16   # or fp32
export PARAKEET_MLX_DECODING=greedy   # or beam
./scripts/run_mlx_macos.sh
```

If you do not want Tailscale exposure, set only localhost:

```bash
export PARAKEET_LISTEN="127.0.0.1:5092"
./scripts/run_mlx_macos.sh
```

For launch-at-login setup, observability, Tailscale endpoint checks, and the
local service menu bar app spec, see [MACOS_SERVICE.md](MACOS_SERVICE.md).

## Usage

### Start the Server

Parakeet TDT provides an OpenAI-compatible API server. By default,
`PARAKEET_BACKEND=auto` uses MLX on Apple Silicon and ONNX elsewhere.

```bash
conda activate parakeet-onnx
export API_KEY="replace-with-strong-key"
python app.py
```
*   **Port**: 5092
*   **Docs**: [http://127.0.0.1:5092/docs](http://127.0.0.1:5092/docs)
*   **Authentication**: `Authorization: Bearer <API_KEY>` is required for transcription and runtime status endpoints.

### Client Example (Python)

You can use the standard `openai` Python library to interact with the server.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5092/v1",
    api_key="replace-with-strong-key"
)

audio_file = open("audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="parakeet-tdt-0.6b-v3",
  file=audio_file,
  response_format="text"
)

print(transcript)
```

### Runtime Selection

The public model name stays `parakeet-tdt-0.6b-v3`. The runtime is selected by
environment variable:

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `PARAKEET_BACKEND` | `auto`, `mlx`, `onnx` | `auto` | `auto` chooses MLX on Apple Silicon and ONNX elsewhere |
| `PARAKEET_MLX_MODEL` | Hugging Face repo | `mlx-community/parakeet-tdt-0.6b-v3` | Used by the MLX backend |
| `PARAKEET_ONNX_MODEL` | Hugging Face repo | `nemo-parakeet-tdt-0.6b-v3` | Used by the ONNX backend |
| `PARAKEET_LISTEN` | space/comma separated `host:port` values | unset | Bind exact interfaces, e.g. localhost plus Tailscale |
| `PARAKEET_MAX_ACTIVE_TRANSCRIPTIONS` | integer | `1` for MLX | Reject extra simultaneous transcriptions with HTTP 429 |
| `PARAKEET_MLX_MEMORY_LIMIT` | byte size, e.g. `24GB` | unset | MLX graph evaluation memory guideline |
| `PARAKEET_MLX_CACHE_LIMIT` | byte size, e.g. `1GB` | unset | MLX free-buffer cache cap; cache is also cleared after each chunk |
| `PARAKEET_MAX_RSS` | byte size, e.g. `32GB` | unset | Process RSS watchdog; exits for launchd restart when exceeded |
| `PARAKEET_MAX_UPLOAD_MB` | number | `2000` | Flask request upload cap in MiB |

The selected model is pre-loaded at startup and cached for subsequent requests.

### Web Interface

The server includes a built-in web interface for testing and easy drag-and-drop transcription.
Access it at: **[http://127.0.0.1:5092](http://127.0.0.1:5092)**

## 🔌 Open WebUI Integration

**This project provides out-of-the-box compatibility with [Open WebUI](https://openwebui.com/)**, serving as a drop-in replacement for OpenAI's speech-to-text API. Experience lightning-fast, local transcription across 25 languages with automatic language detection!

### Setup Instructions

1.  **Start the Parakeet Server** (if not already running):
    ```bash
    conda activate parakeet-onnx
    export API_KEY="replace-with-strong-key"
    python app.py
    ```
    The server will be available at `http://127.0.0.1:5092`

2.  **Configure Open WebUI**:
    - Navigate to **Open WebUI Settings -> Audio**
    - Set **STT Engine** to `OpenAI`
    - Set **OpenAI Base URL** to `http://127.0.0.1:5092/v1`
    - Set **OpenAI API Key** to the same value you set in `API_KEY`
    - Set **STT Model** to `parakeet-tdt-0.6b-v3`
    - Click **Save**

3.  **Start Using Voice!**
    - All voice interactions in Open WebUI will now be transcribed locally
    - Enjoy real-time transcription speeds (up to 30x faster than real-time on modern CPUs)
    - Automatic language detection across all 25 supported languages
    - Complete privacy - all processing happens locally on your machine

## Model details

When running the application, the ONNX models are automatically loaded from the `models/` directory. The primary model used is the **Parakeet TDT 0.6B v3** converted to ONNX with INT8 quantization, providing the optimal balance of speed and accuracy for multilingual speech recognition across 25 European languages.

## 🙏 Acknowledgments

This project stands on the shoulders of giants and wouldn't be possible without:

- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the original FastAPI implementation that served as the foundation for this project
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For developing and open-sourcing the exceptional Parakeet TDT model family
- **[groxaxo](https://github.com/groxaxo)** - The mastermind behind this project, bringing together ONNX optimization, multilingual support, and seamless OpenAI API compatibility

Thank you to all contributors and the open-source community for making high-performance, local speech recognition accessible to everyone!

# Docker Deployment Guide

This document covers Docker deployment options for Parakeet TDT transcription service.

## Quick Start

### CPU Deployment (Recommended for most users)

```bash
# Build and run
export API_KEY="replace-with-strong-key"
docker compose up parakeet-cpu -d

# Or build manually
docker build -f Dockerfile -t parakeet-tdt:cpu .
docker run -d --name parakeet -p 5092:5092 \
    -e API_KEY="${API_KEY}" \
    -v parakeet-models:/app/models parakeet-tdt:cpu
```

### GPU Deployment (Requires NVIDIA GPU)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Build and run with Docker Compose
export API_KEY="replace-with-strong-key"
docker compose up parakeet-gpu -d

# Or build manually
docker build -f Dockerfile.gpu -t parakeet-tdt:gpu .
docker run -d --name parakeet-gpu -p 5092:5092 --gpus all \
    -e API_KEY="${API_KEY}" \
    -v parakeet-models:/app/models parakeet-tdt:gpu
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `http://localhost:5092` | Web UI |
| `http://localhost:5092/health` | Health check (no API key required) |
| `http://localhost:5092/v1/audio/transcriptions` | OpenAI-compatible API (requires `Authorization: Bearer <API_KEY>`) |
| `http://localhost:5092/docs` | Swagger documentation |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(required)* | Shared API key expected in `Authorization: Bearer <API_KEY>` |
| `HF_HOME` | `/app/models` | HuggingFace model cache |
| `HF_HUB_CACHE` | `/app/models` | HuggingFace hub cache |

### Persistent Model Cache

Models are cached in a Docker volume to avoid re-downloading:

```bash
# List volumes
docker volume ls | grep parakeet

# Inspect volume
docker volume inspect parakeet-models

# Remove volume (forces model re-download)
docker volume rm parakeet-models
```

## Files Created

| File | Description |
|------|-------------|
| `Dockerfile` | CPU-only default image (Python 3.10 slim) |
| `Dockerfile.cpu` | CPU-only alternative (same runtime profile) |
| `Dockerfile.gpu` | NVIDIA CUDA 12.1 image with GPU support |
| `docker-compose.yml` | Orchestration for both variants |
| `.dockerignore` | Excludes unnecessary files from build |

## Testing

```bash
# Check health
curl http://localhost:5092/health

# Transcribe audio (OpenAI-compatible)
curl -X POST http://localhost:5092/v1/audio/transcriptions \
    -H "Authorization: Bearer ${API_KEY}" \
    -F "file=@audio.mp3" \
    -F "model=parakeet-tdt-0.6b-v3"
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs parakeet-cpu`
- Ensure `API_KEY` is set before `docker compose up`
- First startup takes ~60s to download the model

**GPU not detected:**
- Verify NVIDIA Container Toolkit: `nvidia-smi` should work inside container
- Run: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

**Out of memory:**
- CPU image requires ~2GB RAM
- GPU image requires ~4GB VRAM

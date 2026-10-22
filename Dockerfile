# Parakeet TDT 0.6B ONNX - CPU Dockerfile
# High-performance speech transcription on CPU

ARG PYTHON_IMAGE=python:3.14.9-slim-trixie
FROM ${PYTHON_IMAGE}

LABEL maintainer="Parakeet TDT"
LABEL description="ONNX-optimized speech transcription service (CPU)"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash parakeet
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements-cpu.lock .

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --cert /etc/ssl/certs/ca-certificates.crt --upgrade "pip>=26.2.2" && \
    pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements-cpu.lock
RUN pip check

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY parakeet.png .

# Create directories for runtime
RUN mkdir -p temp_uploads models && \
    chown -R parakeet:parakeet /app

# Switch to non-root user
USER parakeet

# Environment variables
ENV HF_HOME=/app/models
ENV HF_HUB_CACHE=/app/models
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=true
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 5092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5092/health')" || exit 1

# Run the application
CMD ["python", "app.py"]

# Modal Deployment Guide (L4, scale-to-zero)

This project now includes a Modal-native deployment entrypoint at:

- `modal_app.py`

The Modal deployment is additive. Your existing local Flask app in `app.py` is unchanged.

## What this deployment does

- Preserves OpenAI-compatible REST endpoint: `POST /v1/audio/transcriptions`
- Preserves auth: `Authorization: Bearer <API_KEY>`
- Supports response formats: `json`, `text`, `srt`, `vtt`, `verbose_json`
- Uses `nvidia/parakeet-tdt-0.6b-v3` on Modal GPU
- Uses one GPU container only (`max_containers=1`)
- Uses scale-to-zero (`min_containers=0`) with `scaledown_window=20`
- Uses a Modal Volume at `/cache` for Hugging Face model cache

## Runtime dependencies in Modal image

The image installs:

- `nemo_toolkit[asr]`
- `ffmpeg`
- `pydub`
- `numpy<2`
- `fastapi`
- `python-multipart`

## Prerequisites

1. Install Modal CLI:

```bash
pip install modal
```

2. Authenticate:

```bash
modal setup
```

3. Create API key secret (required by API auth):

```bash
modal secret create parakeet-api-key API_KEY="replace-with-strong-key"
```

## Deploy

From this repo root:

```bash
cd <repo-root>
modal deploy modal_app.py
```

For development hot-reload:

```bash
modal serve modal_app.py
```

## GPU selection

Default GPU is `l4`.

To switch to A10G for benchmarking:

```bash
PARAKEET_MODAL_GPU=a10g modal deploy modal_app.py
```

## Endpoint usage

After deploy, Modal prints your web endpoint URL.

Assume:

```bash
export BASE_URL="https://<your-modal-endpoint>"
export API_KEY="replace-with-strong-key"
```

Health (no auth):

```bash
curl "$BASE_URL/health"
```

Status (auth required):

```bash
curl -H "Authorization: Bearer $API_KEY" "$BASE_URL/status"
```

Transcription:

```bash
curl -X POST "$BASE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@audio.mp3" \
  -F "model=parakeet-tdt-0.6b-v3" \
  -F "response_format=json"
```

## OpenAI client compatibility

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-modal-endpoint>/v1",
    api_key="replace-with-strong-key",
)

with open("audio.mp3", "rb") as f:
    out = client.audio.transcriptions.create(
        model="parakeet-tdt-0.6b-v3",
        file=f,
        response_format="text",
    )

print(out)
```

## Cold start and lag expectations

- With `min_containers=0`, cold starts are expected after idle scale-down.
- `scaledown_window=20` means the container remains warm for 20s after becoming idle.
- Requests arriving while warm avoid model reload and are much faster.
- Model cache in Modal Volume reduces repeat cold-start download overhead.

## Concurrency behavior

- GPU worker runs sequentially (`max_inputs=1`) for predictable memory usage.
- Multiple overlapping requests are accepted and queued by Modal.
- No extra GPU containers are started (`max_containers=1`).

## Cost controls

Code-level controls already configured:

- `min_containers=0`
- `scaledown_window=20`
- `max_containers=1`

Set budget limits in Modal UI:

1. Open your Modal workspace billing/settings.
2. Set a budget alert threshold.
3. Set a hard spending cap if needed.

## Suggested validation checklist

1. Auth check:
   - valid key returns `200`
   - invalid/missing key returns `401`
2. Functional check:
   - short audio returns non-empty transcript
3. Format check:
   - `text`, `srt`, `vtt`, `verbose_json` all return expected structure
4. Queue check:
   - send 2-3 concurrent requests and verify sequential GPU processing
5. Cold/warm latency check:
   - one request after >20s idle (cold)
   - immediate follow-up request (warm)
6. Scale-to-zero check:
   - verify no warm GPU remains after idle window

## Notes

- This Modal path is API-only by design (no HTML UI/templates in v1).
- Local deployment path remains available via existing Docker/Flask files.

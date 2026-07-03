#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.local/share/mise/installs/uv/latest/uv-aarch64-apple-darwin:${PATH:-}"

cd "$(dirname "$0")/.."

if [[ -z "${API_KEY:-}" && -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${API_KEY:-}" ]]; then
  echo "Set API_KEY before launching, for example:"
  echo '  export API_KEY="replace-with-strong-key"'
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with: brew install ffmpeg"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, or run the app with your own Python environment."
  exit 1
fi

PORT="${PORT:-5092}"
VENV="${PARAKEET_VENV:-.venv-mlx}"
TAILSCALE_WAIT_SECONDS="${TAILSCALE_WAIT_SECONDS:-120}"
PARAKEET_REQUIRE_TAILSCALE="${PARAKEET_REQUIRE_TAILSCALE:-0}"
PARAKEET_INSTALL_DEPS="${PARAKEET_INSTALL_DEPS:-auto}"

needs_install=0
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "${PYTHON:-python3}"
  needs_install=1
elif ! "$VENV/bin/python" - <<'PY' >/dev/null 2>&1
import flask
import openai
import parakeet_mlx
import psutil
import requests
import waitress
PY
then
  needs_install=1
fi

if [[ "$PARAKEET_INSTALL_DEPS" == "always" || "$needs_install" == "1" ]]; then
  uv pip install --python "$VENV/bin/python" -r requirements-mlx.txt
fi

tailscale_cli="${TAILSCALE_CLI:-}"
if [[ -z "$tailscale_cli" ]]; then
  if command -v tailscale >/dev/null 2>&1; then
    tailscale_cli="$(command -v tailscale)"
  elif [[ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ]]; then
    tailscale_cli="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
  fi
fi

run_tailscale() {
  TAILSCALE_BE_CLI=1 "$tailscale_cli" "$@"
}

tailscale_ip="${TAILSCALE_IP:-}"
tailscale_dns=""
if [[ -z "$tailscale_ip" && -n "$tailscale_cli" ]]; then
  deadline=$((SECONDS + TAILSCALE_WAIT_SECONDS))
  while true; do
    tailscale_ip="$(run_tailscale ip -4 2>/dev/null | head -n 1 || true)"
    if [[ -n "$tailscale_ip" ]]; then
      break
    fi
    if (( SECONDS >= deadline )); then
      break
    fi
    echo "Waiting for Tailscale IPv4 address..."
    sleep 2
  done
fi

if [[ "$PARAKEET_REQUIRE_TAILSCALE" == "1" && -z "$tailscale_ip" ]]; then
  echo "Tailscale is required, but no Tailscale IPv4 address is available."
  exit 75
fi

if [[ -n "$tailscale_ip" && -n "${TAILSCALE_EXPECTED_IP:-}" && "$tailscale_ip" != "$TAILSCALE_EXPECTED_IP" ]]; then
  echo "Tailscale IP mismatch: expected $TAILSCALE_EXPECTED_IP, got $tailscale_ip"
  exit 76
fi

if [[ -n "$tailscale_cli" ]]; then
  tailscale_dns="$(
    run_tailscale status --json 2>/dev/null \
      | python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get("Self", {}).get("DNSName", "").rstrip("."))' \
      || true
  )"
fi

if [[ -n "$tailscale_dns" && -n "${TAILSCALE_EXPECTED_DNS:-}" && "$tailscale_dns" != "$TAILSCALE_EXPECTED_DNS" ]]; then
  echo "Tailscale DNS mismatch: expected $TAILSCALE_EXPECTED_DNS, got $tailscale_dns"
  exit 77
fi

export PARAKEET_BACKEND="${PARAKEET_BACKEND:-mlx}"
export PARAKEET_MODEL_CACHE="${PARAKEET_MODEL_CACHE:-$HOME/.cache/parakeet-fastapi-openai/models}"
export PARAKEET_OPEN_BROWSER="${PARAKEET_OPEN_BROWSER:-0}"

if [[ -z "${PARAKEET_LISTEN:-}" ]]; then
  if [[ -n "$tailscale_ip" ]]; then
    export PARAKEET_LISTEN="127.0.0.1:${PORT} ${tailscale_ip}:${PORT}"
  else
    export HOST="${HOST:-127.0.0.1}"
  fi
fi

echo "Backend: $PARAKEET_BACKEND"
echo "Cache: $PARAKEET_MODEL_CACHE"
if [[ -n "$tailscale_ip" ]]; then
  echo "Tailscale IP: $tailscale_ip"
fi
if [[ -n "$tailscale_dns" ]]; then
  echo "Tailscale DNS: $tailscale_dns"
fi
if [[ -n "${PARAKEET_LISTEN:-}" ]]; then
  echo "Listen: $PARAKEET_LISTEN"
else
  echo "Listen: ${HOST:-127.0.0.1}:$PORT"
fi

exec "$VENV/bin/python" app.py

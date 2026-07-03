#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$HOME/.local/share/mise/installs/uv/latest/uv-aarch64-apple-darwin:${PATH:-}"

label="${PARAKEET_LAUNCHD_LABEL:-com.parakeet.mlx-server}"
port="${PORT:-5092}"
tailscale_cli="${TAILSCALE_CLI:-}"

if [[ -z "$tailscale_cli" ]]; then
  if command -v tailscale >/dev/null 2>&1; then
    tailscale_cli="$(command -v tailscale)"
  elif [[ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ]]; then
    tailscale_cli="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
  fi
fi

echo "== launchd =="
launchctl print "gui/$(id -u)/$label" 2>/dev/null | sed -n '1,80p' || echo "not loaded"

echo
echo "== listeners =="
lsof -nP -iTCP:"$port" -sTCP:LISTEN || true

echo
echo "== health =="
curl -fsS "http://127.0.0.1:${port}/health" || true
echo

echo
echo "== local https =="
if command -v portless >/dev/null 2>&1; then
  portless list || true
  curl -fsSk "https://parakeet.localhost/health" || true
  echo
else
  echo "portless not found"
fi

if [[ -n "$tailscale_cli" ]]; then
  echo
  echo "== tailscale =="
  TAILSCALE_BE_CLI=1 "$tailscale_cli" ip -4 || true
  TAILSCALE_BE_CLI=1 "$tailscale_cli" status --self || true
  TAILSCALE_BE_CLI=1 "$tailscale_cli" dns status 2>/dev/null | sed -n '1,30p' || true

  echo
  echo "== tailscale serve =="
  TAILSCALE_BE_CLI=1 "$tailscale_cli" serve status --json 2>/dev/null || true
fi

echo
echo "== logs =="
echo "$HOME/Library/Logs/parakeet-mlx-server/stdout.log"
echo "$HOME/Library/Logs/parakeet-mlx-server/stderr.log"

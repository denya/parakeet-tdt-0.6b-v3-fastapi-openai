# macOS Service

The MLX Parakeet server can run as a user LaunchAgent:

- Label: `com.parakeet.mlx-server`
- Plist: `~/Library/LaunchAgents/com.parakeet.mlx-server.plist`
- Launcher: `scripts/run_mlx_macos.sh`
- Runtime: `.venv-mlx`
- Model cache: `~/.cache/parakeet-fastapi-openai/models`
- Logs: `~/Library/Logs/parakeet-mlx-server/`

It is a user LaunchAgent rather than a system LaunchDaemon because this Tailscale
install is often the macOS app/network-extension variant and is normally
available in the user login session. The LaunchAgent starts at login, can wait
for Tailscale, and restarts if the server exits unexpectedly.

## LaunchAgent example

This example starts the service at login and caps local MLX memory pressure:

- `WAITRESS_THREADS=2`
- `PARAKEET_MAX_ACTIVE_TRANSCRIPTIONS=1`
- `PARAKEET_MLX_MEMORY_LIMIT=24GB`
- `PARAKEET_MLX_CACHE_LIMIT=1GB`
- `PARAKEET_MLX_WIRED_LIMIT=24GB`
- `PARAKEET_MAX_RSS=32GB`
- `PARAKEET_MLX_LOCAL_ATTENTION=1`
- `PARAKEET_CHUNK_MINUTE=1.0`
- `PARAKEET_MAX_UPLOAD_MB=512`

`PARAKEET_MLX_MEMORY_LIMIT`, `PARAKEET_MLX_CACHE_LIMIT`, and
`PARAKEET_MLX_WIRED_LIMIT` are MLX/Metal allocator limits. `PARAKEET_MAX_RSS` is
a watchdog limit for ordinary process RSS: if resident memory crosses it, the
server exits with code `75` and launchd restarts it.

Create `~/Library/LaunchAgents/com.parakeet.mlx-server.plist` with the relevant
environment variables for your machine. Keep secrets in the repo `.env` file or
in a local secret store rather than in the plist.

Optional Tailscale hardening:

- `PARAKEET_REQUIRE_TAILSCALE=1`
- `TAILSCALE_EXPECTED_IP=100.x.y.z`
- `TAILSCALE_EXPECTED_DNS=your-mac.your-tailnet.ts.net`

If the machine is removed from the tailnet, Tailscale is reset/reinstalled, or
the node key is lost, Tailscale can assign a new IP. In that case update the
expected values after confirming the new machine identity.

## Service commands

Load or reload:

```bash
label="${PARAKEET_LAUNCHD_LABEL:-com.parakeet.mlx-server}"
plist="$HOME/Library/LaunchAgents/$label.plist"
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$plist"
launchctl kickstart -k "gui/$(id -u)/$label"
```

Stop:

```bash
launchctl bootout "gui/$(id -u)/${PARAKEET_LAUNCHD_LABEL:-com.parakeet.mlx-server}"
```

Status:

```bash
./scripts/service_status.sh
launchctl print "gui/$(id -u)/${PARAKEET_LAUNCHD_LABEL:-com.parakeet.mlx-server}"
lsof -nP -iTCP:5092 -sTCP:LISTEN
curl -fsS http://127.0.0.1:5092/health
```

Logs:

```bash
tail -f "$HOME/Library/Logs/parakeet-mlx-server/stdout.log"
tail -f "$HOME/Library/Logs/parakeet-mlx-server/stderr.log"
```

## Tailscale checks

```bash
TAILSCALE_BE_CLI=1 /Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4
TAILSCALE_BE_CLI=1 /Applications/Tailscale.app/Contents/MacOS/Tailscale status --self
TAILSCALE_BE_CLI=1 /Applications/Tailscale.app/Contents/MacOS/Tailscale dns status
```

MagicDNS is enabled for this tailnet, so clients in the tailnet can use the DNS
name instead of hardcoding the `100.x` address. A stable DNS name can be made by
renaming the machine in the Tailscale admin console or changing the Tailscale
hostname.

## HTTPS

Current operating choice: keep the tailnet API on HTTP until Tailscale HTTPS
certificates are enabled for this tailnet. Traffic between Tailscale nodes is
still protected by Tailscale; the missing part is only browser/API-visible TLS
at the `https://...ts.net` URL.

Portless is configured for local HTTPS:

```bash
portless alias parakeet 5092 --force
portless list
portless service status
curl -fsSk https://parakeet.localhost/health
```

This gives a stable local URL, `https://parakeet.localhost/v1`, through the
Portless proxy on port 443. The Portless proxy is installed as a root
LaunchDaemon at `/Library/LaunchDaemons/sh.portless.proxy.plist`, so the proxy
itself starts after reboot. It is intentionally local: `.localhost` resolves to
loopback and does not expose the API on the Tailscale IP.

For HTTPS on the Tailscale IP and MagicDNS name, use Tailscale Serve rather than
Portless alias routing. The target state is:

```bash
TAILSCALE_BE_CLI=1 /Applications/Tailscale.app/Contents/MacOS/Tailscale \
  serve --bg --https=443 --yes http://127.0.0.1:5092

curl -fsS https://your-mac.your-tailnet.ts.net/health
```

This keeps the Python server bound to local/Tailscale HTTP on `5092`, while
Tailscale terminates HTTPS on the tailnet-facing `443`.

If certificate provisioning fails, enable HTTPS certificates in the Tailscale
admin console under DNS, with MagicDNS enabled. After that, re-run the
`tailscale serve` command above. Portless `--tailscale` also depends on
Tailscale HTTPS certificates and is designed to wrap a child app process, so it
is less suitable than Tailscale Serve for this already-managed LaunchAgent
daemon.

## Menu bar app spec

A small Swift-only helper would be useful if several local model/API services
will run on the same Mac.

Scope:

- SwiftUI `MenuBarExtra`, no Electron.
- Reads service definitions from a local JSON file, for example
  `~/.config/local-services/services.json`.
- For each service, shows launchd state, listening ports, health URL result,
  current PID, uptime, and last log lines.
- Actions: start, stop, restart, open health URL, open logs in Console or
  Terminal, copy local URL, copy Tailscale URL.
- Tailscale panel: current IPv4, MagicDNS name, backend state, MagicDNS enabled,
  expected IP/DNS mismatch warning.
- No secrets in UI. API keys remain in each repo `.env`.

Service definition shape:

```json
{
  "services": [
    {
      "name": "Parakeet MLX",
      "label": "com.parakeet.mlx-server",
      "port": 5092,
      "healthUrl": "http://127.0.0.1:5092/health",
      "localUrl": "http://127.0.0.1:5092/v1",
      "tailscaleUrl": "http://your-mac.your-tailnet.ts.net:5092/v1",
      "stdout": "~/Library/Logs/parakeet-mlx-server/stdout.log",
      "stderr": "~/Library/Logs/parakeet-mlx-server/stderr.log"
    }
  ]
}
```

Implementation notes:

- Use `SMAppService` only for installing/uninstalling helper LaunchAgents if the
  app owns them. For externally managed plists, shell out to `launchctl print`,
  `launchctl kickstart`, and `launchctl bootout`.
- Use `NWConnection` or `URLSession` for health checks.
- Use `OSLogStore` only for unified logging; plain file logs are simpler for
  these Python services.
- Poll every 5-10 seconds while menu is open; slower in background.

Before building this, evaluate existing tools:

- LaunchControl is the strongest general GUI for launchd jobs.
- Lingon X is simpler for editing launchd jobs.
- Ubersicht/SwiftBar/xbar can display status quickly, but they are not a
  service-control app.
- A custom Swift menu bar app makes sense if the goal is one opinionated
  dashboard for local AI/API services plus Tailscale identity.

# CPU Python migration — 22 October 2026

**Candidate: Python 3.14.9 + NumPy 2.5.4 + ONNX Runtime 1.31.0. Real ASR and Docker validation passed. No speedup demonstrated. Target-host A/B and rollout approval remain required.**

## Source of truth and scope

Fetched the repository and inspected `origin/main` (`a69a0fe`), `origin/prod` (`d69910221b86e68f9992e13394e838d6eeee42e9`) and upstream main (`06464cd`). No AGENTS.md or other agent instructions were present in the checkout. The default/deployed branch is **prod**. Main lacks the deployment Dockerfile, API authentication and MLX changes; PR #1 was merged into prod. This PR therefore targets prod and preserves that work. No application/model/framework change is included. `app.py`, `requirements-mlx.txt`, macOS scripts, services and launchd are unchanged. Linux results are not Apple Silicon/MLX validation.

## Supported versions and exact blockers

Official releases as of 2026-10-22:

| Branch | Latest patch | Maintenance status | Decision |
|---|---|---|---|
| 3.10 | 3.10.22 | End of life in October 2026 | Deployed baseline; migrate |
| 3.11 | 3.11.17 | Security fixes until October 2027 | Supported but not newest |
| 3.12 | 3.12.15 | Security fixes until October 2028 | Smallest-change fallback with NumPy 1.26.4 / ORT 1.23.2 |
| 3.13 | 3.13.17 | Security fixes until October 2029 | NumPy 2 required; offers no compatibility advantage over 3.14 |
| 3.14 | 3.14.9 | Bugfix maintenance, security support to October 2030 | Newest compatible candidate |
| 3.15 | 3.15.1 | Current stable, support to October 2031 | Blocked by ONNX Runtime binary availability |

Primary sources: [Python releases](https://www.python.org/downloads/source/), [version list](https://www.python.org/doc/versions/), [branch status](https://devguide.python.org/versions/), [3.14.9 security release](https://www.python.org/downloads/release/python-3149/), [3.15.1](https://www.python.org/downloads/release/python-3151/).

| Package | Metadata / Linux amd64 ABI | Consequence |
|---|---|---|
| NumPy 1.26.4 | Requires-Python >=3.9; officially supports **3.9–3.12**. CPython wheels end at cp312; manylinux_2_17 | `numpy<2` blocks wheel-only installation on 3.13+. The loose python_requires does **not** imply 3.14 source-build support |
| NumPy 2.5.4 | Requires-Python >=3.12; cp312/313/314/315 manylinux_2_27/2_28 wheels (also cp314t/cp315t) | Supports regular 3.14.9 and 3.15.1 |
| ORT 1.23.2 | Requires-Python >=3.10; Linux cp310–cp313 wheels, manylinux_2_27/2_28 | Cannot keep this baseline ORT on 3.14 |
| ORT 1.31.0 | Requires-Python >=3.11; Linux cp311–cp314 wheels, manylinux_2_28 (also cp313t/cp314t) | No cp315 or suitable abi3 wheel; pip binary resolution for 3.15 fails |
| onnx-asr 0.12.0 | Requires-Python >=3.10; py3-none-any; NumPy >=1.22.4; hub extra needs huggingface-hub >=0.30.2 | Pure Python, supports NumPy 1 and 2 / Python through 3.14; no NeMo/PyTorch dependency on this CPU path |

The onnx-asr **cpu extra** excludes ORT 1.24.1, 1.25.*, 1.26.0 and requires ORT <1.24 on Python <3.11. This repository uses `[hub]` plus an explicit ORT dependency, so those extra-specific exclusions would not protect an unconstrained resolver. The candidate pins 1.31.0, outside the excluded releases.

NumPy 2 changes its C ABI. A binary built only for NumPy 1 can fail to load with NumPy 2; removing `<2` alone is insufficient. The exact candidate ORT/NumPy wheels were imported and exercised on real audio. [NumPy 1.26.4 support](https://numpy.org/devdocs/release/1.26.4-notes.html), [NumPy 2 ABI](https://numpy.org/devdocs/release/2.0.0-notes.html), [NumPy metadata](https://pypi.org/pypi/numpy/2.5.4/json), [ORT baseline metadata](https://pypi.org/pypi/onnxruntime/1.23.2/json), [ORT candidate metadata](https://pypi.org/pypi/onnxruntime/1.31.0/json), [onnx-asr metadata](https://pypi.org/pypi/onnx-asr/0.12.0/json), [ORT Python 3.15 issue](https://github.com/microsoft/onnxruntime/issues/33271).

No NumPy source compilation was attempted. Regular GIL CPython was tested; JIT and free-threaded builds were not selected. Some cp314t wheels exist, but that does not validate this application's entire concurrency/dependency stack.

## Before / after

| Component | Supplied Hetzner baseline | Tested candidate |
|---|---|---|
| Python | 3.10.22 | 3.14.9 |
| NumPy | 1.26.4 | 2.5.4 |
| onnxruntime | 1.23.2 | 1.31.0 |
| onnx-asr | 0.12.0 | 0.12.0 |
| Flask / Waitress | 3.1.3 / 3.0.2 | 3.1.3 / 3.0.2 |
| certifi / pip | 2026.10.19 / 26.2.2 | 2026.10.19 / 26.2.2 |
| Image | python:3.10-slim | python:3.14.9-slim-trixie |
| Model | Parakeet v3 CPU INT8 | Same model, INT8, CPU provider |

Full candidate package versions and selected wheel SHA-256 values are in `requirements-cpu.lock` and `results/wheel-integrity.json`. This lock deliberately targets **Linux amd64 CPython 3.14**, not all OSes/architectures. Docker installs with `--only-binary=:all: --require-hashes` and runs `pip check`. `requirements.txt` preserves NumPy 1 for Python <3.13 and allows NumPy 2 for newer interpreters; the Docker lock controls the server build. The legacy Dockerfile.cpu is unchanged; deploy this candidate with **Dockerfile**.

## Measurements: cloud, not Hetzner

AMD EPYC 9V74, Linux x86_64, fixed affinity CPUs 0–3; enclosing cloud cgroup 8 CPU quota / 8 GiB memory, identical across environments. No per-process memory cgroup could be created here. CPython builds for the three reported runs were python-build-standalone, not the initial scratch source compilation. Same host ffmpeg 6.1.1, ORT intra/inter threads 4/1, sequential execution, all graph optimizations, CPUExecutionProvider only, Waitress 4 threads, two active transcription slots. The model snapshot was pinned to `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`; hashes are in `results/model-hashes.json`. Cache was shared and offline during timing.

Five public recordings, 72.56 seconds total: EN JFK plus RU/ES FLEURS. Twenty warm repetitions each, **100 requests per variant**. First request and one warmup per file excluded from warm percentiles. p95 is the nearest-rank observation. RSS is Linux `ru_maxrss` of the service, including the later codec/long/concurrency checks, not whole-container RSS. Startup is a **new process with model files already cached**, not download time or disk-cold startup. OS page cache was not flushed. Single ordered run per variant; small differences can include cloud noise and cannot prove a causal Python speedup.

| Metric | 3.10.22 / NumPy 1.26.4 / ORT 1.23.2 | 3.12.15 / same NumPy & ORT | 3.14.9 / NumPy 2.5.4 / ORT 1.31.0 |
|---|---:|---:|---:|
| Process startup to /health, cached model | 1.827 s | 2.887 s | 3.446 s |
| First request | 0.764 s | 0.731 s | 0.705 s |
| Warm p50 | 0.822 s | 0.838 s | 0.854 s |
| Warm p95 | 1.349 s | 1.385 s | 1.437 s |
| Median RTF (lower better) | 0.0624 | 0.0624 | 0.0646 |
| Sequential audio seconds / wall second | 15.973 | 16.002 | 15.541 |
| Concurrency 2, requests/s (JFK, 8 requests) | 1.501 | 1.588 | 1.412 |
| Peak process RSS | 2076.7 MiB | 2119.4 MiB | 2268.2 MiB |
| Normalized WER | 7/142 = 4.93% | 7/142 = 4.93% | 6/142 = 4.23% |
| Normalized CER | 18/850 = 2.12% | 18/850 = 2.12% | 17/850 = 2.00% |

Raw samples, transcripts, installed versions and checks: `results/cloud-{310,312,314}-full.json`. These reproduce the supplied core baseline, **not the exact private Hetzner image**: the unreported transitive baseline versions were resolved here. Candidate differences also include filelock and httpx2/httpcore2 versions; full snapshots disclose them. No claim that these numbers predict Hetzner latency.

**Interpretation:** candidate p50 +3.9%, p95 +6.6%, RSS +9.2% in this run. Do not sell this as acceleration. 3.12 with unchanged numeric libraries was essentially flat in aggregate throughput. Upgrade value is supported Python and a tested binary dependency set.

### Text preservation

3.10 and 3.12 produced identical texts on all five fixtures. Candidate changed one EN comma and corrected RU `пустыне` to `пустыни`; other fixture text was identical. All repetitions of each fixture were stable within a variant. `1990` and the geographical name `Китае/Китая` were retained. The FLEURS ES reference ends with `Oravec, 2002`; **all variants omit that tail**. This is not a new migration regression, but does not establish preservation of arbitrary names/numbers. WER/CER lowercases, applies NFKC and removes punctuation, with spaces retained for CER. It does not equate written numbers with spoken words. This tiny corpus is a regression fixture, not a production accuracy estimate.

### Bottleneck

Separate instrumented diagnostic run (not mixed into latency percentiles): 37 recognition calls took 42.082 accumulated seconds; ORT encoder calls accounted for 39.308 s (93.4%), decoder/joint calls 2.038 s (4.8%). The remaining 0.736 s (1.8%) includes file loading, NumPy preprocessing and Python decoder orchestration. ffmpeg took 3.146 s over 40 subprocess calls and ffprobe 2.057 s over 36 calls. Concurrent timings are summed per call and may overlap; they are not exclusive CPU time. Native ONNX inference dominates the measured recognition path. No evidence makes CPython execution the primary bottleneck. Short-clip latency also includes subprocess/I/O cost; concurrency shares CPU and increases latency. `results/cloud-314-profile.json` contains the measurements.

## Functional validation and existing limitations

Real Flask/Waitress HTTP requests passed on all three environments: `/health`, OpenAPI JSON, missing/wrong/correct bearer auth, missing file rejection, whisper-1 fallback, json/text/verbose_json/SRT/VTT, FLAC/MP3/M4A/Opus OGG/Opus WebM/stereo 44.1 kHz WAV conversion, invalid audio rejection, 100-second chunked input, concurrent requests and temporary-file cleanup. The actual OpenAI SDK is checked in an additional run and CI. No paid API or private audio was used.

Do not overstate compatibility: language/prompt/temperature/timestamp granularity options are accepted but not implemented; verbose_json hardcodes `english`, words are not a faithful OpenAI word-timestamp implementation, and malformed audio currently returns 500. These are existing application behaviors; this Python migration does not change them. Codec checks verify successful nonempty transcription; exact lossy-codec texts are recorded rather than assumed identical. No load/soak test or exhaustive timeout/cancellation test was performed.

Docker was unavailable in the Work cloud (no daemon, no namespace permission). GitHub Actions successfully built the actual image and passed imports, ffmpeg, pip check, HTTPS control and all real ASR/API/codec tests with 4 CPU / 6 GiB limits and networking disabled during transcription. [Docker + real ASR run](https://github.com/denya/parakeet-tdt-0.6b-v3-fastapi-openai/actions/runs/39783908108). `results/ci-docker-314.json` records that separate machine; its latencies must not be compared against this cloud baseline.

## Security and integrity

- PR #1's system ca-certificates refresh, secure pip bootstrap via system CA (pinned to the fixed 26.2.2 release), certifi security floor and strict TLS remain. No insecure-host flag, verify=False or unverified SSL context was introduced.
- pip-audit 2.10.1 checked all 40 pinned candidate environment packages (including pip): **zero known advisories** at collection time; raw output in `results/pip-audit.json`. PyPI per-version vulnerability lists were also checked for downloaded wheels. This is not a guarantee against unknown compromise or OS vulnerabilities.
- 39 runtime wheels were downloaded using the hash lock and independently SHA-256 matched to PyPI metadata. No selected file was yanked; no `.pth` file was found. Full file identities, dates, python_requires and hashes are retained. onnx-asr and ORT PyPI releases have no publisher provenance attestations; matching PyPI hashes proves download integrity, not independent reproducible-build provenance.
- Checked official package release/security pages and supply-chain reports. No specific compromise of these selected versions was identified. The relevant CA incident is addressed by certifi 2026.10.19 and pip 26.2.2; no claim that general absence of advisories certifies a package as safe.
- CI actions are pinned to commit SHAs, read-only contents permission, and checkout does not persist credentials. Test API key is synthetic. Model download is pinned to a repository revision and recorded file hashes. No remote Python model code is enabled.
- The local cloud intercepts HTTPS: an explicit certifi-only control could not trust its proxy CA, so **that local negative TLS test is inconclusive**, not a pass. `check_tls.py` runs in CI with a valid HTTPS control plus invalid certificates for both system and certifi trust stores. The exact twelve incident-specific chains from PR #1 have not been re-tested here.
- OS package/CVE scan, Hetzner's trust configuration and network path remain rollout checks. First CI build resolved python image manifest `sha256:b347e2dcbc7f0b6f9fb7182e9da6a187fb10ae715e0692518c1aed0eec94a700`, ffmpeg 7.1.5-0+deb13u1. Base tag and apt packages can change on rebuild; record the final deployed image ID.

## Reproduce and target-host gate

Build the newest compatible candidate (this does not deploy):

```sh
docker build --pull -f Dockerfile -t parakeet:python314-candidate .
```

Run on the authorized Hetzner host, after checking headroom. This script starts temporary containers, publishes no ports, never calls Coolify, never reads production secrets and uses its own pinned model cache. It runs A-B-B-A with identical 4 CPU / 6 GiB limits, model and fixtures. Override BENCH_CPUSET if CPU IDs 0–3 are unavailable. A busy production host will contaminate timing; use a quiet maintenance window without stopping production from this script.

```sh
bash scripts/python-upgrade/run_docker_ab.sh \
  l8okos4wogs0kc88o4w8s80c:d69910221b86e68f9992e13394e838d6eeee42e9 \
  parakeet:python314-candidate
```

No SSH/Hetzner access was assumed or attempted through the user's Mac. Unverified on target: exact image/transitive dependencies, CPU instruction set/performance, filesystem cache, cgroup limits, Coolify environment, TLS/CA path, real concurrency distribution, proxy upload limits, long-duration memory behavior. The runner tests mounted unchanged app.py with each image's runtime; inspect image IDs/package snapshots and compare model hashes before drawing conclusions.

For the minimal Python-only fallback, use Python 3.12.15 with NumPy 1.26.4 and ORT 1.23.2. Its real API results are included, but it retains NumPy 1's old line and is **not the selected Docker deployment**. Merely overriding PYTHON_IMAGE to 3.12 will not work with the cp314-specific candidate lock; it needs a separately resolved cp312 wheel lock.

## Rollout / rollback (not executed)

1. Review this PR targeting prod; preserve baseline image and current Coolify configuration. Do not merge if prod auto-deploy would violate the current research-only instruction.
2. Run target A/B. All auth/API checks must pass, no unexplained normalized-text regression, no OOM. Review p95/throughput/RSS against the target's actual SLO/headroom. Proposed investigation thresholds: >10% p95 regression, >15% RSS increase or any new transcript regression; these are gates to discuss, not guarantees.
3. With explicit rollout approval, deploy the reviewed candidate image to resource `l8okos4wogs0kc88o4w8s80c`, retaining ports, secrets, model, storage and CPU settings. Check public `/health`, then an authorized public-fixture transcription and logs. Observe latency/error rate/RSS before accepting the migration.
4. On failure, redeploy preserved image `l8okos4wogs0kc88o4w8s80c:d69910221b86e68f9992e13394e838d6eeee42e9` through the existing Coolify workflow and recheck health/auth/transcription. There is no schema/state migration to undo. Do not rebuild the old tag from floating dependencies as a substitute for the preserved image.

## Updating the lock

`candidate-versions.txt` is the exact resolved environment (pip included). For Linux amd64 CPython 3.14, download those packages with pip `--only-binary=:all:` into an empty wheel directory, verify each filename/hash against the official PyPI version JSON, and write one `name==version --hash=sha256:...` entry per selected runtime wheel (pip remains bootstrapped in Dockerfile). Check dependencies with `pip check`, run pip-audit, rebuild, rerun the fixtures and refresh the integrity report. Never substitute an sdist to make a wheel-only check succeed.

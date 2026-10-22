#!/usr/bin/env bash
# Isolated side-by-side validation only: never changes Coolify, production ports or model cache.
set -euo pipefail
base_image=${1:?Usage: run_docker_ab.sh BASE_IMAGE CANDIDATE_IMAGE [OUTPUT_DIR]}
candidate_image=${2:?Candidate image required}
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
output_dir=${3:-"$repo_dir/benchmark_results/python-upgrade"}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
cache_dir=$(mktemp -d)
trap 'rm -rf -- "$cache_dir"' EXIT
# Match the non-root user of both images, without touching the service's own directories.
chmod 777 "$cache_dir" "$output_dir"
cpus=${BENCH_CPUS:-4}
cpuset=${BENCH_CPUSET:-0-3}
memory=${BENCH_MEMORY:-6g}
rounds=${BENCH_ROUNDS:-20}
docker image inspect "$base_image" "$candidate_image" > "$output_dir/images.json"
lscpu > "$output_dir/lscpu.txt"
docker run --rm --cpus="$cpus" --memory="$memory" --entrypoint python \
  -v "$repo_dir:/src:ro" -v "$cache_dir:/cache" "$candidate_image" \
  /src/scripts/python-upgrade/prepare_model.py --cache /cache > "$output_dir/model.json"
# A-B-B-A exposes drift and repeats process-cold starts; OS page cache is deliberately not flushed.
for label in baseline-1 candidate-1 candidate-2 baseline-2; do
  image=$candidate_image
  if [[ $label == baseline-* ]]; then image=$base_image; fi
  docker run --rm --network=none --cpus="$cpus" --cpuset-cpus="$cpuset" --memory="$memory" --entrypoint python \
    -v "$repo_dir:/src:ro" -v "$cache_dir:/cache:ro" -v "$output_dir:/results" "$image" \
    /src/scripts/python-upgrade/benchmark.py --manifest /src/tests/fixtures/python-upgrade/manifest.json \
    --cache /cache --output "/results/$label.json" --rounds "$rounds"
done
printf 'Results: %s\n' "$output_dir"

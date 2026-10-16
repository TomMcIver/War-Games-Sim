#!/usr/bin/env bash
# Run any command inside the Rust build container (host linking is broken).
# Usage: bash infra/run.sh cargo test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ROOT="$(cd "$ROOT" && pwd -W)"; export MSYS_NO_PATHCONV=1;; esac
IMAGE="${WARSIM_IMAGE:-docker.io/library/rust:1-slim-bookworm}"
exec podman run --rm -v "$ROOT:/work" -v warsim-cargo:/usr/local/cargo/registry \
  -e CARGO_TARGET_DIR=/tmp/target -w /work "$IMAGE" "$@"

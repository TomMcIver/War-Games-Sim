#!/usr/bin/env bash
# Run any command inside the dev container (host Rust linking is broken here).
# Build the image once:  bash infra/build-image.sh
# Usage:  bash infra/run.sh cargo test
#         bash infra/run.sh bash -c "maturin develop --release && pytest python"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ROOT="$(cd "$ROOT" && pwd -W)"; export MSYS_NO_PATHCONV=1;; esac
IMAGE="${WARSIM_IMAGE:-localhost/warsim-dev}"
# Cache the cargo target dir per checkout so builds are incremental.
KEY="$(basename "$ROOT" | tr -c 'A-Za-z0-9_.-' '_')"
exec podman run --rm -v "$ROOT:/work" \
  -v warsim-cargo:/usr/local/cargo/registry \
  -v "warsim-target-$KEY:/cache/target" \
  -e CARGO_TARGET_DIR=/cache/target -w /work "$IMAGE" "$@"

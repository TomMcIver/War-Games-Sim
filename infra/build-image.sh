#!/usr/bin/env bash
# Build the shared dev image (Rust + Python + maturin + torch CPU).
set -euo pipefail
cd "$(dirname "$0")"
exec podman build -t localhost/warsim-dev -f Containerfile .

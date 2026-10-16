#!/usr/bin/env bash
# Run the full check suite in the dev container, like CI does:
#   cargo fmt --check, clippy -D warnings, cargo test, build sim-py + pytest
#   (bindings and rl tests, when present), plus the viewer's Node tests on the
#   host if `node` is installed (they need no packages).
#
# Usage: bash infra/test-all.sh [--rust-only]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
warsim_require_podman

status=0
bash "$HERE/run.sh" bash infra/checks.sh "$@" || status=1

if [ "${1:-}" != "--rust-only" ]; then
  printf '\n\033[1m==> viewer tests (host node)\033[0m\n'
  if command -v node >/dev/null 2>&1; then
    (cd "$HERE/.." && node --test viewer/tests/*.test.js) || status=1
  else
    echo "--  skip: node not installed (CI runs them)"
  fi
fi
exit "$status"

#!/usr/bin/env bash
# Serve the replay viewer (and training replays) from the dev container.
#
# Usage:  bash infra/viewer.sh [port]          (default 8000 or $WARSIM_VIEWER_PORT)
# Then open:
#   http://localhost:PORT/viewer/                         bundled sample
#   http://localhost:PORT/viewer/?replay=/runs/<path>.json   a replay from runs/
#   http://localhost:PORT/runs/                           browse the runs volume
#
# viewer/ and runs/ are mounted read-only; the port is bound to 127.0.0.1 only.
# runs/ is the `warsim-runs` volume used by infra/train.sh (or $WARSIM_RUNS_DIR).
# Stop with Ctrl+C (or: podman stop warsim-viewer).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

PORT="${1:-${WARSIM_VIEWER_PORT:-8000}}"
case "$PORT" in ''|*[!0-9]*) warsim_die "port must be a number (got '$PORT')" ;; esac
warsim_require_podman
RUNS="$(warsim_runs_source)"

echo "viewer:  http://localhost:$PORT/viewer/"
echo "replays: http://localhost:$PORT/viewer/?replay=/runs/<file>.json  (list: http://localhost:$PORT/runs/)"
exec podman run --rm --init --replace --name warsim-viewer \
  -p "127.0.0.1:$PORT:8000" \
  -v "$WARSIM_ROOT/viewer:/site/viewer:ro" \
  -v "$RUNS:/site/runs:ro" \
  "$WARSIM_IMAGE" python3 -m http.server 8000 --bind 0.0.0.0 --directory /site

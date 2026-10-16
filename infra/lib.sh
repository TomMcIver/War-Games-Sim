#!/usr/bin/env bash
# Shared settings for infra/*.sh. Source it, don't run it:
#   . "$(dirname "$0")/lib.sh"
# Works from Linux/macOS bash and Git Bash on Windows (Podman needs a Windows
# path for bind mounts, and MSYS must not rewrite /work-style container paths).

WARSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) WARSIM_ROOT="$(cd "$WARSIM_ROOT" && pwd -W)"; export MSYS_NO_PATHCONV=1 ;;
esac
WARSIM_IMAGE="${WARSIM_IMAGE:-localhost/warsim-dev}"
# Per-checkout cargo target volume (same naming as infra/run.sh).
WARSIM_KEY="$(basename "$WARSIM_ROOT" | tr -c 'A-Za-z0-9_.-' '_')"

warsim_die() { echo "error: $*" >&2; exit 1; }

# Source for the runs/ mount: the named volume `warsim-runs` by default, or a
# host directory when WARSIM_RUNS_DIR is set (created if missing).
warsim_runs_source() {
  if [ -z "${WARSIM_RUNS_DIR:-}" ]; then
    echo "warsim-runs"
    return
  fi
  mkdir -p "$WARSIM_RUNS_DIR" || warsim_die "cannot create WARSIM_RUNS_DIR=$WARSIM_RUNS_DIR"
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) (cd "$WARSIM_RUNS_DIR" && pwd -W) ;;
    *) (cd "$WARSIM_RUNS_DIR" && pwd) ;;
  esac
}

warsim_require_podman() {
  command -v podman >/dev/null 2>&1 || warsim_die "podman not found (see docs/SETUP.md)"
  podman image exists "$WARSIM_IMAGE" \
    || warsim_die "image $WARSIM_IMAGE missing - build it once with: bash infra/build-image.sh"
}

# Mount/env flags shared by every container we start (same as infra/run.sh).
# shellcheck disable=SC2034  # used by the scripts that source this file
WARSIM_COMMON_ARGS=(
  -v "$WARSIM_ROOT:/work"
  -v warsim-cargo:/usr/local/cargo/registry
  -v "warsim-target-$WARSIM_KEY:/cache/target"
  -e CARGO_TARGET_DIR=/cache/target
  -w /work
)

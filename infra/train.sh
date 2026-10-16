#!/usr/bin/env bash
# Train in the dev container: builds the `warsim` bindings (incremental, cached
# target dir), then runs the RL trainer with the repo mounted at /work.
# Output goes to a persistent volume mounted at /runs (survives --rm):
# the named volume `warsim-runs`, or a host directory if WARSIM_RUNS_DIR is set.
#
# Usage:  bash infra/train.sh [trainer args...]
#   bash infra/train.sh --config mock_small --name exp1
#   bash infra/train.sh --config warsim_small --set env.n_envs=64 --resume auto
#   WARSIM_DETACH=1 bash infra/train.sh --config warsim_small   # background;
#       follow with: podman logs -f warsim-train
#   WARSIM_RUNS_DIR=./runs bash infra/train.sh ...              # host folder
# Args go to `python -m warsim_rl.train` (see python/warsim_rl/train.py);
# `--out-dir /runs` is added unless you pass --out-dir yourself.
# Override the entry module with WARSIM_TRAIN_MODULE (default warsim_rl.train).
set -euo pipefail
. "$(dirname "$0")/lib.sh"
warsim_require_podman

MODULE="${WARSIM_TRAIN_MODULE:-warsim_rl.train}"
RUNS="$(warsim_runs_source)"
ARGS=("$@")
case " $* " in *" --out-dir"*) ;; *) ARGS+=(--out-dir /runs) ;; esac

MODE=(--rm)
if [ "${WARSIM_DETACH:-0}" = 1 ]; then
  MODE=(-d --replace --name warsim-train)
  echo "training in background: podman logs -f warsim-train   (stop: podman stop warsim-train)"
fi

# Inside the container: build+install the bindings if they exist, then train.
INNER='set -e
if [ -f crates/sim-py/pyproject.toml ]; then
  (cd crates/sim-py && maturin develop --release --quiet)
fi
export PYTHONPATH="/work/python${PYTHONPATH:+:$PYTHONPATH}"
exec python -m "$0" "$@"'

exec podman run "${MODE[@]}" --init "${WARSIM_COMMON_ARGS[@]}" \
  -v "$RUNS:/runs" \
  -e PYTHONUNBUFFERED=1 \
  "$WARSIM_IMAGE" bash -c "$INNER" "$MODULE" "${ARGS[@]}"

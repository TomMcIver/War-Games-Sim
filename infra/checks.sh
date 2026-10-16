#!/usr/bin/env bash
# Every check we gate merges on. Runs wherever the toolchain is installed:
# inside the dev image (via infra/test-all.sh) or directly on a CI runner.
# Needs: cargo + rustfmt + clippy; for the Python parts an active venv with
# maturin, pytest, numpy and torch (CPU). Pieces that have not landed yet
# (crates/sim-py, python/tests_*) are skipped, not failed.
#
# Usage: bash infra/checks.sh [--rust-only]
# Runs every step even if one fails, then exits non-zero if any failed.
set -uo pipefail
cd "$(dirname "$0")/.."

RUST_ONLY=0
case "${1:-}" in
  --rust-only) RUST_ONLY=1 ;;
  "") ;;
  *) echo "usage: bash infra/checks.sh [--rust-only]" >&2; exit 2 ;;
esac

FAILED=()
SKIPPED=()
step() {
  local name="$1"; shift
  printf '\n\033[1m==> %s\033[0m\n' "$name"
  if "$@"; then
    printf '\033[32mok\033[0m  %s\n' "$name"
  else
    printf '\033[31mFAIL\033[0m %s\n' "$name"
    FAILED+=("$name")
    return 1
  fi
}
skip() { printf '\n--  skip %s (%s)\n' "$1" "$2"; SKIPPED+=("$1"); }

step "cargo fmt --check" cargo fmt --all -- --check || true
step "cargo clippy -D warnings" cargo clippy --workspace --all-targets -- -D warnings || true
step "cargo test" cargo test --workspace || true

build_sim_py() { (cd crates/sim-py && maturin develop --release); }
rl_tests() { (cd python && PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m pytest tests_rl -q); }

if [ "$RUST_ONLY" = 1 ]; then
  skip "python" "--rust-only"
else
  HAVE_WARSIM=0
  if [ -f crates/sim-py/pyproject.toml ]; then
    if step "build sim-py (maturin develop)" build_sim_py; then HAVE_WARSIM=1; fi
  else
    skip "sim-py" "crates/sim-py not present"
  fi
  if [ -d python/tests_bindings ] && [ "$HAVE_WARSIM" = 1 ]; then
    step "pytest python/tests_bindings" python -m pytest python/tests_bindings -q || true
  else
    skip "binding tests" "python/tests_bindings or the warsim module missing"
  fi
  if [ -d python/tests_rl ]; then
    step "pytest python/tests_rl" rl_tests || true
  else
    skip "rl tests" "python/tests_rl not present"
  fi
fi

echo
if [ "${#SKIPPED[@]}" -gt 0 ]; then echo "skipped: ${SKIPPED[*]}"; fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\033[31mFAILED:\033[0m %s\n' "${FAILED[*]}"
  exit 1
fi
printf '\033[32mall checks passed\033[0m\n'

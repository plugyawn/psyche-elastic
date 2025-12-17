#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer an explicit PYO3_PYTHON, otherwise use the repo venv, otherwise fall back to python3.
PYTHON="${PYO3_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "[psyche-env] Could not find a usable python (tried: ${PYTHON})." >&2
  exit 1
fi

# Locate libtorch via python's torch install.
TORCH_LIB="$("${PYTHON}" - <<'PY'
import os, sys
try:
    import torch
except Exception as e:
    print(f"[psyche-env] Failed to import torch: {e}", file=sys.stderr)
    sys.exit(1)
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PYO3_PYTHON="${PYTHON}"

case "$(uname -s)" in
  Darwin)
    PY_LIBDIR="$("${PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"
    if [[ -n "${PY_LIBDIR}" ]]; then
      export DYLD_LIBRARY_PATH="${PY_LIBDIR}:${DYLD_LIBRARY_PATH:-}"
      export LIBRARY_PATH="${PY_LIBDIR}:${LIBRARY_PATH:-}"
    fi
    export DYLD_LIBRARY_PATH="${TORCH_LIB}:${DYLD_LIBRARY_PATH:-}"
    ;;
  *)
    export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
    ;;
esac

# Keep build artifacts local to the repo by default (helps multi-machine reproducibility).
export CARGO_HOME="${CARGO_HOME:-${ROOT}/.cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${ROOT}/.cargo-target}"

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

RUNTIME_DIR="${ROOT}/python-runtime"
if [[ -f "${RUNTIME_DIR}/requirements.lock.txt" ]]; then
  REQUIREMENTS="${RUNTIME_DIR}/requirements.lock.txt"
elif [[ -f "${ROOT}/packaging/python-runtime/requirements.lock.txt" ]]; then
  REQUIREMENTS="${ROOT}/packaging/python-runtime/requirements.lock.txt"
else
  echo "[bootstrap] Could not find requirements.lock.txt." >&2
  exit 1
fi

VENV_DIR="${ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ -x "${PYTHON_BIN}" ]]; then
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
import safetensors.torch
PY
  then
    echo "[bootstrap] ${VENV_DIR} already OK."
    exit 0
  fi
fi

echo "[bootstrap] Creating ${VENV_DIR} and installing runtime deps..."

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT}/.uv-cache}"

if command -v uv >/dev/null 2>&1; then
  PYTHON="$(uv python find 3.12 --no-project --no-config 2>/dev/null || true)"
  if [[ -z "${PYTHON}" ]]; then
    echo "[bootstrap] Python 3.12 not found. Install it with:" >&2
    echo "  uv python install 3.12" >&2
    exit 1
  fi
  uv venv "${VENV_DIR}" --python "${PYTHON}" --seed pip >/dev/null
  uv pip install --python "${PYTHON_BIN}" --requirements "${REQUIREMENTS}"
else
  PYTHON_SYS=""
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_SYS="$(command -v python3.12)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_SYS="$(command -v python3)"
  else
    echo "[bootstrap] python3 not found; install Python 3.12 (or install uv)." >&2
    exit 1
  fi

  if ! "${PYTHON_SYS}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 2)
PY
  then
    echo "[bootstrap] Python 3.12 is required for the pinned runtime deps." >&2
    echo "[bootstrap] Install it with either:" >&2
    echo "  - uv:  brew install uv && uv python install 3.12" >&2
    echo "  - python.org installer for 3.12 (or Homebrew python@3.12), then re-run this script" >&2
    exit 1
  fi

  "${PYTHON_SYS}" -m venv "${VENV_DIR}"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS}"
fi

echo "[bootstrap] Done."

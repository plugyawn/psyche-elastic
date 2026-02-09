#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NUM_CHUNKS="${NUM_CHUNKS:-1}"
OUT_DIR="${OUT_DIR:-data/fineweb10B}"
BASE_CKPT="${BASE_CKPT:-checkpoints/nanogpt-20m-init}"
TIER_CKPT="${TIER_CKPT:-checkpoints/nanogpt-20m-init-tier1}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Prepare local assets for NanoGPT FineWeb runs on a single A100 host.

Environment overrides:
  NUM_CHUNKS   Number of FineWeb train chunks to download (default: 1)
  OUT_DIR      Dataset output directory (default: data/fineweb10B)
  BASE_CKPT    Base checkpoint path (default: checkpoints/nanogpt-20m-init)
  TIER_CKPT    Tier-1 checkpoint path (default: checkpoints/nanogpt-20m-init-tier1)
EOF
  exit 0
fi

# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

PYTHON_BIN="${PYO3_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "[prepare] Using python: ${PYTHON_BIN}"
echo "[prepare] Bootstrapping runtime..."
bash "${ROOT}/scripts/bootstrap-python-runtime.sh"

if [[ ! -d "${BASE_CKPT}" ]]; then
  echo "[prepare] Base checkpoint not found at ${BASE_CKPT}; creating NanoGPT init checkpoint..."
  "${PYTHON_BIN}" "${ROOT}/scripts/create_small_nanogpt.py"
  mkdir -p "$(dirname "${BASE_CKPT}")"
  rm -rf "${BASE_CKPT}"
  cp -R /tmp/nanogpt-20m-init "${BASE_CKPT}"
fi

if [[ ! -d "${TIER_CKPT}" ]]; then
  echo "[prepare] Tier-1 checkpoint not found at ${TIER_CKPT}; exporting tier slices..."
  "${PYTHON_BIN}" "${ROOT}/scripts/export_matformer_tiers.py" \
    --src "${BASE_CKPT}" \
    --tiers 1
fi

echo "[prepare] Downloading FineWeb data to ${OUT_DIR} (NUM_CHUNKS=${NUM_CHUNKS})..."
"${PYTHON_BIN}" "${ROOT}/scripts/download_fineweb10B.py" \
  --out-dir "${OUT_DIR}" \
  --num-chunks "${NUM_CHUNKS}"

echo "[prepare] Done."
echo "[prepare] Assets ready:"
echo "  - ${BASE_CKPT}"
echo "  - ${TIER_CKPT}"
echo "  - ${OUT_DIR}"

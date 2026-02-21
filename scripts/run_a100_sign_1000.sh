#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT}/config/test-nanogpt-20m-fineweb-1000-distro/state.toml}"
SERVER_PORT="${SERVER_PORT:-21032}"
ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260215}"
DEVICE="${DEVICE:-cuda}"
TIERS="${TIERS:-0,1}"
NUM_CLIENTS="${NUM_CLIENTS:-2}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/runs/$(date -u +%Y%m%d_%H%M%S)_a100_sign1000}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Run a 1000-step sign/sign NanoGPT FineWeb experiment on a single A100 machine.

Environment overrides:
  CONFIG_PATH          Run state TOML path
  SERVER_PORT          Centralized server port (default: 21032)
  ALIGNED_BATCH_SEED   Seed for aligned batches (default: 20260215)
  DEVICE               Client device string (default: cuda)
  TIERS                Comma-separated matformer tiers (default: 0,1)
  NUM_CLIENTS          Number of clients to spawn (default: 2)
  LOG_DIR              Output directory for server/client logs
  SKIP_BUILD           Set to 1 to skip release build preflight
EOF
  exit 0
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[run] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[run] Building release binaries (server + client)..."
  cargo build --release -p psyche-centralized-server -p psyche-centralized-client
fi

mkdir -p "${LOG_DIR}"
SERVER_CONSOLE_LOG="${LOG_DIR}/server.console.log"
SERVER_JSON_LOG="${LOG_DIR}/server.jsonl"
CLIENTS_CONSOLE_LOG="${LOG_DIR}/clients.console.log"

RUN_ID="$(
  python3 - <<'PY' "${CONFIG_PATH}"
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
path = sys.argv[1]
with open(path, "rb") as f:
    cfg = tomllib.load(f)
print(cfg["run_id"])
PY
)"

wait_for_server() {
  local pid="$1"
  local port="$2"
  local retries=180

  while (( retries > 0 )); do
    if python3 - <<'PY' "${port}"
import socket
import sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.4)
ok = s.connect_ex(("127.0.0.1", port)) == 0
s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[run] Server exited before becoming ready." >&2
      return 1
    fi

    sleep 1
    retries=$((retries - 1))
  done

  echo "[run] Timed out waiting for server on port ${port}." >&2
  return 1
}

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run] Starting server..."
(
  RUST_LOG="warn,psyche_centralized_server=info" \
    bash "${ROOT}/scripts/psyche-centralized-server.sh" run \
      --state "${CONFIG_PATH}" \
      --server-port "${SERVER_PORT}" \
      --tui false \
      --logs json \
      --write-log "${SERVER_JSON_LOG}" \
      --aligned-batches \
      --aligned-batches-seed "${ALIGNED_BATCH_SEED}"
) >"${SERVER_CONSOLE_LOG}" 2>&1 &
SERVER_PID=$!

wait_for_server "${SERVER_PID}" "${SERVER_PORT}"

echo "[run] Starting clients..."
(
  RUST_LOG="warn,psyche_client=info,psyche_centralized_client=info,psyche_modeling=info" \
    bash "${ROOT}/scripts/psyche-centralized-spawn-clients.sh" \
      --run-id "${RUN_ID}" \
      --server-addr "127.0.0.1:${SERVER_PORT}" \
      --num-clients "${NUM_CLIENTS}" \
      --device "${DEVICE}" \
      --tiers "${TIERS}" \
      --logs json \
      --log-dir "${LOG_DIR}" \
      --rust-log "warn,psyche_client=info,psyche_centralized_client=info,psyche_modeling=info" \
      -- \
      --matformer-helper-fraction 0 \
      --matformer-helper-rotation-interval 16 \
      --distro-apply-mode sign \
      --distro-value-mode sign
) >"${CLIENTS_CONSOLE_LOG}" 2>&1

echo "[run] Clients finished."
echo "[run] Logs written to: ${LOG_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PATH="${HOME}/.cargo/bin:${PATH}"

# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

if [[ -f "${ROOT}/env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/env.local"
  set +a
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[lsolo] WANDB_API_KEY is not set. Put it in env.local on the host." >&2
  exit 1
fi

require_a100() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[lsolo] nvidia-smi not found; this script must be run on a CUDA A100 host." >&2
    exit 1
  fi
  if ! nvidia-smi -L | grep -q "A100"; then
    echo "[lsolo] Expected an A100 GPU but nvidia-smi -L did not report one." >&2
    nvidia-smi -L || true
    exit 1
  fi
  "${PYO3_PYTHON:-python3}" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[lsolo] Failed to import torch in python preflight: {e}", file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("[lsolo] torch.cuda.is_available() is false; CUDA torch is required on the A100 host.", file=sys.stderr)
    raise SystemExit(1)
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    print(f"[lsolo] Expected CUDA device 0 to be A100, got: {name}", file=sys.stderr)
    raise SystemExit(1)
PY
}
require_a100

CFG_PATH="${CFG_PATH:-config/test-nanogpt-20m-fineweb-2000-lsolo-fast/state.toml}"
ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260216}"

SERVER_PORT="${SERVER_PORT:-23110}"
METRICS_PORT="${METRICS_PORT:-6290}"
P2P_PORT="${P2P_PORT:-33120}"
DEVICE="${DEVICE:-cuda}"

IROH_DISCOVERY="${IROH_DISCOVERY:-local}"
IROH_RELAY="${IROH_RELAY:-disabled}"

DISTRO_APPLY_MODE="${DISTRO_APPLY_MODE:-sign}"
DISTRO_VALUE_MODE="${DISTRO_VALUE_MODE:-sign}"

WANDB_PROJECT="${WANDB_PROJECT:-psyche-a100}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-a100-2000-lsolo-$(date -u +%Y%m%d_%H%M%S)}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/a100_lsolo_2000/$(date -u +%Y%m%d_%H%M%S)}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-14400}" # 4h hard cap
mkdir -p "${LOG_ROOT}"

echo "[lsolo] cfg=${CFG_PATH}"
echo "[lsolo] log_root=${LOG_ROOT}"
echo "[lsolo] wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_GROUP}"

echo "[lsolo] Building release binaries (server + client)..."
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

PY="${PYO3_PYTHON:-python3}"
read -r RUN_ID TOTAL_STEPS < <("${PY}" - "${CFG_PATH}" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["run_id"], cfg["config"]["total_steps"])
PY
)

export RUST_LOG="warn,psyche_client=info,psyche_modeling=info,psyche_centralized_server=info,psyche_centralized_client=info"

SERVER_JSON="${LOG_ROOT}/server.jsonl"
SERVER_CONSOLE="${LOG_ROOT}/server.console.log"

CLIENT_JSON="${LOG_ROOT}/client-tier0.jsonl"
CLIENT_CONSOLE="${LOG_ROOT}/client-tier0.console.log"

# Use a unique identity by default (override via RAW_IDENTITY_SECRET_KEY).
export RAW_IDENTITY_SECRET_KEY="${RAW_IDENTITY_SECRET_KEY:-$(printf '%064x' 13000)}"

"${CARGO_TARGET_DIR}/release/psyche-centralized-server" run \
  --state "${CFG_PATH}" \
  --server-port "${SERVER_PORT}" \
  --tui false \
  --logs json \
  --write-log "${SERVER_JSON}" \
  --aligned-batches \
  --aligned-batches-seed "${ALIGNED_BATCH_SEED}" \
  >"${SERVER_CONSOLE}" 2>&1 &
SERVER_PID=$!

# Wait for the TCP server port.
retries=240
while (( retries > 0 )); do
  if "${PY}" - <<'PY' "${SERVER_PORT}"
import socket,sys
port=int(sys.argv[1])
s=socket.socket(); s.settimeout(0.4)
ok=s.connect_ex(("127.0.0.1", port))==0
s.close()
raise SystemExit(0 if ok else 1)
PY
  then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[lsolo] Server exited before becoming ready (port=${SERVER_PORT})." >&2
    exit 1
  fi
  sleep 1
  retries=$((retries - 1))
done

if (( retries == 0 )); then
  echo "[lsolo] Timed out waiting for server on port ${SERVER_PORT}." >&2
  exit 1
fi

METRICS_LOCAL_PORT="${METRICS_PORT}" \
  "${CARGO_TARGET_DIR}/release/psyche-centralized-client" train \
  --run-id "${RUN_ID}" \
  --server-addr "127.0.0.1:${SERVER_PORT}" \
  --device "${DEVICE}" \
  --matformer-tier 0 \
  --bind-p2p-port "${P2P_PORT}" \
  --iroh-discovery "${IROH_DISCOVERY}" \
  --iroh-relay "${IROH_RELAY}" \
  --logs json \
  --write-log "${CLIENT_JSON}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
  --wandb-run "${WANDB_GROUP}-lsolo-tier0" \
  --matformer-helper-fraction 0 \
  --matformer-helper-rotation-interval 16 \
  --distro-apply-mode "${DISTRO_APPLY_MODE}" \
  --distro-value-mode "${DISTRO_VALUE_MODE}" \
  >"${CLIENT_CONSOLE}" 2>&1 &
CLIENT_PID=$!

has_loss_at_step() {
  local path="$1"
  local target_step="$2"
  "${PY}" - "$path" "$target_step" <<'PY'
import sys

path = sys.argv[1]
target = str(int(sys.argv[2]))

try:
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")
            is_loss = ("integration_test_log_marker=loss" in line) or ('"integration_test_log_marker":"loss"' in line)
            has_step = (f"step={target}" in line) or (f'"step":{target}' in line)
            if is_loss and has_step:
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

has_finished_at_step() {
  local path="$1"
  local target_step="$2"
  "${PY}" - "$path" "$target_step" <<'PY'
import sys

path = sys.argv[1]
target = str(int(sys.argv[2]))

try:
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")

            is_state = ("integration_test_log_marker=state_change" in line) or ('"integration_test_log_marker":"state_change"' in line)
            is_finished = ("new_state=Finished" in line) or ('"new_state":"Finished"' in line)
            has_step = (f"step={target}" in line) or (f'"step":{target}' in line)
            if is_state and is_finished and has_step:
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

TRAIN_LAST_STEP=$((TOTAL_STEPS - 1))
echo "[lsolo] Waiting for completion (train_last_step=${TRAIN_LAST_STEP}, total_steps=${TOTAL_STEPS})..."
start_ts="$(date +%s)"
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[lsolo] Server died unexpectedly; see ${SERVER_CONSOLE}" >&2
    exit 1
  fi
  if ! kill -0 "${CLIENT_PID}" 2>/dev/null; then
    echo "[lsolo] Client died unexpectedly; see ${CLIENT_CONSOLE}" >&2
    exit 1
  fi
  if has_loss_at_step "${CLIENT_JSON}" "${TRAIN_LAST_STEP}" || has_finished_at_step "${CLIENT_JSON}" "${TOTAL_STEPS}"; then
    echo "[lsolo] Reached completion; shutting down..."
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts > MAX_WAIT_SECS )); then
    echo "[lsolo] Timed out waiting for step ${TOTAL_STEPS} after ${MAX_WAIT_SECS}s; shutting down." >&2
    exit 1
  fi
  sleep 10
done

kill "${CLIENT_PID}" >/dev/null 2>&1 || true
wait "${CLIENT_PID}" >/dev/null 2>&1 || true

kill "${SERVER_PID}" >/dev/null 2>&1 || true
wait "${SERVER_PID}" >/dev/null 2>&1 || true

echo "[lsolo] done"

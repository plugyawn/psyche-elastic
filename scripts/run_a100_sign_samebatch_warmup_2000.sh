#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PATH="${HOME}/.cargo/bin:${PATH}"

# 2000-step sign/sign NanoGPT FineWeb run, but with an early "same-batch warmup" where
# all trainers fetch the same canonical batch for the first N steps.
#
# This is an *ablation/debugging* tool to encourage early gradient agreement between tiers.
#
# NOTE: `--same-batch-warmup-steps` aliases data client-side (BatchId provenance is not preserved).
# Do not use this for on-chain runs.

CFG_PATH="${CFG_PATH:-config/test-nanogpt-20m-fineweb-2000-distro-fast/state.toml}"
ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260216}"

SERVER_PORT="${SERVER_PORT:-23120}"
METRICS_BASE_PORT="${METRICS_BASE_PORT:-6310}"
P2P_BASE_PORT="${P2P_BASE_PORT:-33220}"
DEVICE="${DEVICE:-cuda}"

IROH_DISCOVERY="${IROH_DISCOVERY:-local}"
IROH_RELAY="${IROH_RELAY:-disabled}"

SAME_BATCH_WARMUP_STEPS="${SAME_BATCH_WARMUP_STEPS:-500}"

WANDB_PROJECT="${WANDB_PROJECT:-psyche-a100}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-a100-2000-sign-samebatch${SAME_BATCH_WARMUP_STEPS}-$(date -u +%Y%m%d_%H%M%S)}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/a100_sign_samebatch_2000/$(date -u +%Y%m%d_%H%M%S)}"
SKIP_BUILD="${SKIP_BUILD:-0}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-14400}" # 4h hard cap

usage() {
  cat <<'EOF'
Run a 2000-step sign/sign NanoGPT FineWeb experiment on a single A100 host, with
client-side same-batch warmup for early steps.

Environment overrides:
  CFG_PATH                state.toml path (default: config/test-nanogpt-20m-fineweb-2000-distro-fast/state.toml)
  ALIGNED_BATCH_SEED      seed for --aligned-batches (default: 20260216)
  SERVER_PORT             centralized server port (default: 23120)
  METRICS_BASE_PORT       base port for per-client metrics (default: 6310)
  P2P_BASE_PORT           base port for per-client iroh bind (default: 33220)
  DEVICE                  client device string (default: cuda)
  IROH_DISCOVERY          iroh discovery mode (default: local)
  IROH_RELAY              iroh relay kind (default: disabled)
  SAME_BATCH_WARMUP_STEPS initial steps to alias to a shared batch (default: 500; 0 disables)
  WANDB_PROJECT           wandb project (default: psyche-a100)
  WANDB_ENTITY            wandb entity (default: unset)
  WANDB_GROUP             wandb group (default: derived)
  LOG_ROOT                output directory (default: logs/a100_sign_samebatch_2000/<timestamp>/)
  SKIP_BUILD              set to 1 to skip release build preflight
  MAX_WAIT_SECS           hard timeout for run completion (default: 14400)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[run] Config not found: ${CFG_PATH}" >&2
  exit 1
fi

# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

if [[ -f "${ROOT}/env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/env.local"
  set +a
fi

require_a100() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run] nvidia-smi not found; this script must be run on a CUDA A100 host." >&2
    exit 1
  fi
  if ! nvidia-smi -L | grep -q "A100"; then
    echo "[run] Expected an A100 GPU but nvidia-smi -L did not report one." >&2
    nvidia-smi -L || true
    exit 1
  fi

  "${PYO3_PYTHON:-python3}" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[run] Failed to import torch in python preflight: {e}", file=sys.stderr)
    raise SystemExit(1)
if not torch.cuda.is_available():
    print("[run] torch.cuda.is_available() is false; CUDA torch is required on the A100 host.", file=sys.stderr)
    raise SystemExit(1)
name = torch.cuda.get_device_name(0)
if "A100" not in name:
    print(f"[run] Expected CUDA device 0 to be A100, got: {name}", file=sys.stderr)
    raise SystemExit(1)
PY
}

require_a100

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[run] Building release binaries (server + client)..."
  cargo build --release -p psyche-centralized-server -p psyche-centralized-client
fi

mkdir -p "${LOG_ROOT}"
echo "[run] cfg=${CFG_PATH}"
echo "[run] log_root=${LOG_ROOT}"
echo "[run] wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_GROUP}"

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

SERVER_JSON="${LOG_ROOT}/server.jsonl"
SERVER_CONSOLE="${LOG_ROOT}/server.console.log"

server_pid=""
client_pids=()

cleanup() {
  for pid in "${client_pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
  for pid in "${client_pids[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  if [[ -n "${server_pid}" ]]; then
    wait "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run] Starting server (port=${SERVER_PORT})..."
RUST_LOG="warn,psyche_centralized_server=info" \
  "${CARGO_TARGET_DIR}/release/psyche-centralized-server" run \
    --state "${CFG_PATH}" \
    --server-port "${SERVER_PORT}" \
    --tui false \
    --logs json \
    --write-log "${SERVER_JSON}" \
    --aligned-batches \
    --aligned-batches-seed "${ALIGNED_BATCH_SEED}" \
    >"${SERVER_CONSOLE}" 2>&1 &
server_pid=$!

# Wait for server port bind.
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
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[run] Server exited before becoming ready (port=${SERVER_PORT})." >&2
    exit 1
  fi
  sleep 1
  retries=$((retries - 1))
done
if (( retries == 0 )); then
  echo "[run] Timed out waiting for server on port ${SERVER_PORT}." >&2
  exit 1
fi

export RUST_LOG="warn,psyche_client=info,psyche_modeling=info,psyche_centralized_client=info"

for tier in 0 1; do
  key_int=$((14000 + tier))
  raw_key="$(printf '%064x' "${key_int}")"
  metrics_port=$((METRICS_BASE_PORT + tier))
  p2p_port=$((P2P_BASE_PORT + tier))

  client_json="${LOG_ROOT}/client-tier${tier}.jsonl"
  client_console="${LOG_ROOT}/client-tier${tier}.console.log"

  wandb_args=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb_args+=(--wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    wandb_args+=(--wandb-run "${WANDB_GROUP}-samebatch-tier${tier}")
  fi

  echo "[run] Starting client tier=${tier} metrics_port=${metrics_port} p2p_port=${p2p_port}"
  RAW_IDENTITY_SECRET_KEY="${raw_key}" METRICS_LOCAL_PORT="${metrics_port}" \
    "${CARGO_TARGET_DIR}/release/psyche-centralized-client" train \
      --run-id "${RUN_ID}" \
      --server-addr "127.0.0.1:${SERVER_PORT}" \
      --device "${DEVICE}" \
      --matformer-tier "${tier}" \
      --bind-p2p-port "${p2p_port}" \
      --iroh-discovery "${IROH_DISCOVERY}" \
      --iroh-relay "${IROH_RELAY}" \
      --logs json \
      --write-log "${client_json}" \
      "${wandb_args[@]}" \
      --matformer-helper-fraction 0 \
      --matformer-helper-rotation-interval 16 \
      --distro-apply-mode sign \
      --distro-value-mode sign \
      --same-batch-warmup-steps "${SAME_BATCH_WARMUP_STEPS}" \
      >"${client_console}" 2>&1 &
  client_pids+=("$!")
done

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

            # Support both "key=value" logs and JSON logs.
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
echo "[run] Waiting for completion (train_last_step=${TRAIN_LAST_STEP}, total_steps=${TOTAL_STEPS})..."
start_ts="$(date +%s)"
while true; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[run] Server died unexpectedly; see ${SERVER_CONSOLE}" >&2
    exit 1
  fi
  for pid in "${client_pids[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[run] A client died unexpectedly; see ${LOG_ROOT}/client-*.console.log" >&2
      exit 1
    fi
  done

  if (has_loss_at_step "${LOG_ROOT}/client-tier0.jsonl" "${TRAIN_LAST_STEP}" || has_finished_at_step "${LOG_ROOT}/client-tier0.jsonl" "${TOTAL_STEPS}") \
    && (has_loss_at_step "${LOG_ROOT}/client-tier1.jsonl" "${TRAIN_LAST_STEP}" || has_finished_at_step "${LOG_ROOT}/client-tier1.jsonl" "${TOTAL_STEPS}")
  then
    echo "[run] Reached completion; shutting down..."
    break
  fi

  now_ts="$(date +%s)"
  if (( now_ts - start_ts > MAX_WAIT_SECS )); then
    echo "[run] Timed out waiting for step ${TOTAL_STEPS} after ${MAX_WAIT_SECS}s; shutting down." >&2
    exit 1
  fi

  sleep 10
done

echo "[run] done"

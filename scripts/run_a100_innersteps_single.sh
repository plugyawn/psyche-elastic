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

CFG_PATH="${CFG_PATH:-config/exp-ngpt20m-fw-5000-ls-b400/state.toml}"
ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260220}"
DEVICE="${DEVICE:-cuda}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
IROH_DISCOVERY="${IROH_DISCOVERY:-local}"
IROH_RELAY="${IROH_RELAY:-disabled}"
HELDOUT_EVAL_BATCHES="${HELDOUT_EVAL_BATCHES:-32}"
HELDOUT_EVAL_BATCH_SIZE="${HELDOUT_EVAL_BATCH_SIZE:-16}"
TIER1_INNER_STEPS="${TIER1_INNER_STEPS:-4}"

SERVER_PORT="${SERVER_PORT:-23620}"
METRICS_BASE_PORT="${METRICS_BASE_PORT:-7110}"
P2P_BASE_PORT="${P2P_BASE_PORT:-36120}"
CLIENT_START_DELAY_SECS="${CLIENT_START_DELAY_SECS:-3}"

WANDB_PROJECT="${WANDB_PROJECT:-psyche-a100}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-a100-innersteps-single-$(date -u +%Y%m%d_%H%M%S)}"
WANDB_STEP_LOGGING="${WANDB_STEP_LOGGING:-1}"

LOG_ROOT="${LOG_ROOT:-}"
SKIP_BUILD="${SKIP_BUILD:-0}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-86400}"
COSINE_SHADOW="${COSINE_SHADOW:-1}"

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[inner] Config not found: ${CFG_PATH}" >&2
  exit 1
fi

require_a100() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[inner] nvidia-smi not found; this script must run on CUDA host." >&2
    exit 1
  fi
  if ! nvidia-smi -L | grep -q "A100"; then
    echo "[inner] Expected A100 GPU." >&2
    nvidia-smi -L || true
    exit 1
  fi
}
require_a100

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[inner] Building release binaries..."
  cargo build --release -p psyche-centralized-server -p psyche-centralized-client
fi

RUN_ID="$(
  awk -F'=' '
    /^run_id[[:space:]]*=/ {
      val=$2
      gsub(/"/, "", val)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' "${CFG_PATH}"
)"

TOTAL_STEPS="$(
  awk -F'=' '
    /^\[config\]$/ { in_config=1; next }
    /^\[/ { if (in_config) exit }
    in_config && /^total_steps[[:space:]]*=/ {
      val=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' "${CFG_PATH}"
)"

if [[ -z "${RUN_ID}" || -z "${TOTAL_STEPS}" ]]; then
  echo "[inner] Failed to parse run_id/total_steps from ${CFG_PATH}" >&2
  exit 1
fi

if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT}/logs/a100_innersteps_single/${RUN_ID}/$(date -u +%Y%m%d_%H%M%S)"
fi
mkdir -p "${LOG_ROOT}"

SERVER_JSON="${LOG_ROOT}/server.jsonl"
SERVER_CONSOLE="${LOG_ROOT}/server.console.log"

echo "[inner] cfg=${CFG_PATH}"
echo "[inner] run_id=${RUN_ID} total_steps=${TOTAL_STEPS}"
echo "[inner] log_root=${LOG_ROOT}"
echo "[inner] tier1_inner_steps=${TIER1_INNER_STEPS}"
echo "[inner] heldout_eval_batches=${HELDOUT_EVAL_BATCHES} heldout_eval_batch_size=${HELDOUT_EVAL_BATCH_SIZE}"
echo "[inner] client_start_delay_secs=${CLIENT_START_DELAY_SECS}"
echo "[inner] wandb_group=${WANDB_GROUP}"

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

echo "[inner] Starting server (port=${SERVER_PORT})..."
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

retries=240
while (( retries > 0 )); do
  if python3 - <<'PY' "${SERVER_PORT}"
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
    echo "[inner] Server exited before ready." >&2
    exit 1
  fi
  sleep 1
  retries=$((retries - 1))
done
if (( retries == 0 )); then
  echo "[inner] Timed out waiting for server port ${SERVER_PORT}." >&2
  exit 1
fi

if (( CLIENT_START_DELAY_SECS > 0 )); then
  sleep "${CLIENT_START_DELAY_SECS}"
fi

export RUST_LOG="warn,psyche_client=info,psyche_modeling=info,psyche_centralized_client=info"
if [[ "${COSINE_SHADOW}" == "1" ]]; then
  export PSYCHE_DISTRO_COSINE_MIXER_SHADOW=1
fi

for tier in 0 1; do
  key_int=$((18000 + tier))
  raw_key="$(printf '%064x' "${key_int}")"
  metrics_port=$((METRICS_BASE_PORT + tier))
  p2p_port=$((P2P_BASE_PORT + tier))
  client_json="${LOG_ROOT}/client-tier${tier}.jsonl"
  client_console="${LOG_ROOT}/client-tier${tier}.console.log"

  inner_steps=1
  if [[ "${tier}" == "1" ]]; then
    inner_steps="${TIER1_INNER_STEPS}"
  fi

  wandb_args=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb_args+=(--wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}")
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
      wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    wandb_args+=(--wandb-run "${WANDB_GROUP}-tier${tier}")
    if [[ "${WANDB_STEP_LOGGING}" == "1" ]]; then
      wandb_args+=(--wandb-step-logging)
    fi
  fi

  echo "[inner] Starting client tier=${tier} inner_steps=${inner_steps}"
  RAW_IDENTITY_SECRET_KEY="${raw_key}" METRICS_LOCAL_PORT="${metrics_port}" \
    "${CARGO_TARGET_DIR}/release/psyche-centralized-client" train \
      --run-id "${RUN_ID}" \
      --server-addr "127.0.0.1:${SERVER_PORT}" \
      --device "${DEVICE}" \
      --micro-batch-size "${MICRO_BATCH_SIZE}" \
      --matformer-tier "${tier}" \
      --matformer-local-inner-steps "${inner_steps}" \
      --matformer-helper-fraction 0 \
      --matformer-helper-rotation-interval 16 \
      --bind-p2p-port "${p2p_port}" \
      --iroh-discovery "${IROH_DISCOVERY}" \
      --iroh-relay "${IROH_RELAY}" \
      --distro-apply-mode sign \
      --distro-value-mode raw \
      --heldout-eval-batches "${HELDOUT_EVAL_BATCHES}" \
      --heldout-eval-batch-size "${HELDOUT_EVAL_BATCH_SIZE}" \
      --logs json \
      --write-log "${client_json}" \
      "${wandb_args[@]}" \
      >"${client_console}" 2>&1 &

  client_pids+=("$!")
done

has_loss_at_step() {
  local path="$1"
  local target_step="$2"
  python3 - "$path" "$target_step" <<'PY'
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

train_last_step=$((TOTAL_STEPS - 1))
start_ts=$(date +%s)
echo "[inner] Waiting for completion (train_last_step=${train_last_step})..."
while true; do
  now=$(date +%s)
  elapsed=$((now - start_ts))
  if (( elapsed > MAX_WAIT_SECS )); then
    echo "[inner] Timed out after ${elapsed}s." >&2
    exit 1
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[inner] Server exited early." >&2
    exit 1
  fi
  for pid in "${client_pids[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[inner] Client exited early pid=${pid}." >&2
      exit 1
    fi
  done
  ok=1
  for tier in 0 1; do
    if ! has_loss_at_step "${LOG_ROOT}/client-tier${tier}.console.log" "${train_last_step}"; then
      ok=0
      break
    fi
  done
  if (( ok == 1 )); then
    break
  fi
  sleep 15
done

echo "[inner] Reached completion; shutting down..."

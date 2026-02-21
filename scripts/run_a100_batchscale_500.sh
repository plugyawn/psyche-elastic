#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PATH="${HOME}/.cargo/bin:${PATH}"

# Batch-scaling experiment runner (A100).
#
# This script runs a 500-step NanoGPT-20m FineWeb experiment with:
#   --distro-apply-mode sign
#   --distro-value-mode raw
#
# It auto-selects how many clients to spawn based on the config's `min_clients`:
#   min_clients=1 -> tier0 only (L-solo)
#   min_clients=2 -> tier0+tier1 (L+S)
#
# Use `MICRO_BATCH_SIZE` to avoid pathological "micro_batch_size=1 => 200 microbatches"
# slowdowns for large global batches.

CFG_PATH="${CFG_PATH:-config/exp-ngpt20m-fw-500-lsolo-b200/state.toml}"
ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260217}"

SERVER_PORT="${SERVER_PORT:-23320}"
METRICS_BASE_PORT="${METRICS_BASE_PORT:-6510}"
P2P_BASE_PORT="${P2P_BASE_PORT:-34320}"
DEVICE="${DEVICE:-cuda}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-100}"

DISTRO_APPLY_MODE="${DISTRO_APPLY_MODE:-sign}"
DISTRO_VALUE_MODE="${DISTRO_VALUE_MODE:-raw}"
DISTRO_AGGREGATE_MODE="${DISTRO_AGGREGATE_MODE:-legacy}"
DISTRO_EXTRA_ARGS="${DISTRO_EXTRA_ARGS:-}"

IROH_DISCOVERY="${IROH_DISCOVERY:-local}"
IROH_RELAY="${IROH_RELAY:-disabled}"

# Optional: tier-0 suffix gate schedule (disabled by default).
MATFORMER_SUFFIX_GATE_WARMUP_STEPS="${MATFORMER_SUFFIX_GATE_WARMUP_STEPS:-0}"
MATFORMER_SUFFIX_GATE_START_STEP="${MATFORMER_SUFFIX_GATE_START_STEP:-0}"
MATFORMER_SUFFIX_GATE_TIER="${MATFORMER_SUFFIX_GATE_TIER:-1}"
MATFORMER_SUFFIX_GATE_SCHEDULE="${MATFORMER_SUFFIX_GATE_SCHEDULE:-linear}"

# If 1, set PSYCHE_DISTRO_COSINE_MIXER_SHADOW=1 for clients (logs "would drop" stats only).
COSINE_SHADOW="${COSINE_SHADOW:-1}"
# Held-out eval (runs in cooldown and logs integration_test_log_marker=heldout_eval).
HELDOUT_EVAL_BATCHES="${HELDOUT_EVAL_BATCHES:-32}"
HELDOUT_EVAL_BATCH_SIZE="${HELDOUT_EVAL_BATCH_SIZE:-16}"

WANDB_PROJECT="${WANDB_PROJECT:-psyche-a100}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-a100-batchscale-500-$(date -u +%Y%m%d_%H%M%S)}"

LOG_ROOT="${LOG_ROOT:-}"
SKIP_BUILD="${SKIP_BUILD:-0}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-21600}" # 6h hard cap
WAIT_FOR_MODEL_EXTRACT="${WAIT_FOR_MODEL_EXTRACT:-0}"
WAIT_FOR_MODEL_EXTRACT_TIMEOUT_SECS="${WAIT_FOR_MODEL_EXTRACT_TIMEOUT_SECS:-600}"

usage() {
  cat <<'EOF'
Run a 500-step A100 batch-scaling experiment for NanoGPT-20m FineWeb.

Environment overrides:
  CFG_PATH           state.toml path (default: config/exp-ngpt20m-fw-500-lsolo-b200/state.toml)
  ALIGNED_BATCH_SEED seed for --aligned-batches
  SERVER_PORT        centralized server port
  METRICS_BASE_PORT  base port for per-client metrics
  P2P_BASE_PORT      base port for per-client iroh bind
  DEVICE             client device string (default: cuda)
	  MICRO_BATCH_SIZE   per-step microbatch size (default: 100)
	  IROH_DISCOVERY     iroh discovery mode (default: local)
	  IROH_RELAY         iroh relay kind (default: disabled)
	  MATFORMER_SUFFIX_GATE_WARMUP_STEPS  tier-0 suffix gate warmup steps (0=disabled)
	  MATFORMER_SUFFIX_GATE_START_STEP    tier-0 suffix gate start step
	  MATFORMER_SUFFIX_GATE_TIER          which tier defines the "core" width (default: 1)
	  MATFORMER_SUFFIX_GATE_SCHEDULE      linear|cosine (default: linear)
	  COSINE_SHADOW      if 1, set PSYCHE_DISTRO_COSINE_MIXER_SHADOW=1 for clients (default: 1)
	  HELDOUT_EVAL_BATCHES     held-out eval batches at cooldown (default: 32)
	  HELDOUT_EVAL_BATCH_SIZE  held-out eval batch size (default: 16)
	  WANDB_PROJECT      wandb project
	  WANDB_ENTITY       wandb entity
	  WANDB_GROUP        wandb group
  LOG_ROOT           output directory (default: logs/a100_batchscale_500/<run_id>/<timestamp>)
  SKIP_BUILD         set to 1 to skip release build preflight
  MAX_WAIT_SECS      hard timeout (default: 21600)
  WAIT_FOR_MODEL_EXTRACT if 1, wait for clients to enter cooldown and extract the full model (default: 0)
  WAIT_FOR_MODEL_EXTRACT_TIMEOUT_SECS timeout for cooldown extract wait (default: 600)

Notes:
  - DisTrO mode knobs (defaults match current baseline):
      --distro-apply-mode "${DISTRO_APPLY_MODE}" --distro-value-mode "${DISTRO_VALUE_MODE}" --distro-aggregate-mode "${DISTRO_AGGREGATE_MODE}"
  - Optional extra client args via DISTRO_EXTRA_ARGS.
  - It spawns 1 or 2 clients based on config.config.min_clients.
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

PY="${PYO3_PYTHON:-python3}"
read -r RUN_ID TOTAL_STEPS MIN_CLIENTS < <("${PY}" - "${CFG_PATH}" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["run_id"], cfg["config"]["total_steps"], cfg["config"]["min_clients"])
PY
)

if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT}/logs/a100_batchscale_500/${RUN_ID}/$(date -u +%Y%m%d_%H%M%S)"
fi

mkdir -p "${LOG_ROOT}"
echo "[run] cfg=${CFG_PATH}"
echo "[run] run_id=${RUN_ID} total_steps=${TOTAL_STEPS} min_clients=${MIN_CLIENTS}"
echo "[run] log_root=${LOG_ROOT}"
echo "[run] micro_batch_size=${MICRO_BATCH_SIZE}"
echo "[run] distro_apply_mode=${DISTRO_APPLY_MODE} distro_value_mode=${DISTRO_VALUE_MODE} distro_aggregate_mode=${DISTRO_AGGREGATE_MODE}"
echo "[run] distro_extra_args=${DISTRO_EXTRA_ARGS}"
echo "[run] suffix_gate_warmup_steps=${MATFORMER_SUFFIX_GATE_WARMUP_STEPS} start_step=${MATFORMER_SUFFIX_GATE_START_STEP} gate_tier=${MATFORMER_SUFFIX_GATE_TIER} schedule=${MATFORMER_SUFFIX_GATE_SCHEDULE}"
echo "[run] heldout_eval_batches=${HELDOUT_EVAL_BATCHES} heldout_eval_batch_size=${HELDOUT_EVAL_BATCH_SIZE}"
echo "[run] wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_GROUP}"

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
if [[ "${COSINE_SHADOW}" == "1" ]]; then
  export PSYCHE_DISTRO_COSINE_MIXER_SHADOW=1
fi

tiers=()
case "${MIN_CLIENTS}" in
  1) tiers=(0) ;;
  2) tiers=(0 1) ;;
  *)
    echo "[run] Unsupported min_clients=${MIN_CLIENTS}; this script expects 1 or 2." >&2
    exit 1
    ;;
esac

for tier in "${tiers[@]}"; do
  key_int=$((16000 + tier))
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
    wandb_args+=(--wandb-run "${WANDB_GROUP}-tier${tier}")
	  fi
	  suffix_gate_args=()
	  if [[ "${tier}" == "0" && "${MATFORMER_SUFFIX_GATE_WARMUP_STEPS}" != "0" ]]; then
	    suffix_gate_args+=(--matformer-suffix-gate-warmup-steps "${MATFORMER_SUFFIX_GATE_WARMUP_STEPS}")
	    suffix_gate_args+=(--matformer-suffix-gate-start-step "${MATFORMER_SUFFIX_GATE_START_STEP}")
	    suffix_gate_args+=(--matformer-suffix-gate-tier "${MATFORMER_SUFFIX_GATE_TIER}")
	    suffix_gate_args+=(--matformer-suffix-gate-schedule "${MATFORMER_SUFFIX_GATE_SCHEDULE}")
	  fi

	  echo "[run] Starting client tier=${tier} metrics_port=${metrics_port} p2p_port=${p2p_port}"
	  RAW_IDENTITY_SECRET_KEY="${raw_key}" METRICS_LOCAL_PORT="${metrics_port}" \
	    "${CARGO_TARGET_DIR}/release/psyche-centralized-client" train \
      --run-id "${RUN_ID}" \
      --server-addr "127.0.0.1:${SERVER_PORT}" \
      --device "${DEVICE}" \
      --micro-batch-size "${MICRO_BATCH_SIZE}" \
      --matformer-tier "${tier}" \
      --matformer-helper-fraction 0 \
      --matformer-helper-rotation-interval 16 \
      --bind-p2p-port "${p2p_port}" \
      --iroh-discovery "${IROH_DISCOVERY}" \
      --iroh-relay "${IROH_RELAY}" \
      --distro-apply-mode "${DISTRO_APPLY_MODE}" \
      --distro-value-mode "${DISTRO_VALUE_MODE}" \
      --distro-aggregate-mode "${DISTRO_AGGREGATE_MODE}" \
      --heldout-eval-batches "${HELDOUT_EVAL_BATCHES}" \
      --heldout-eval-batch-size "${HELDOUT_EVAL_BATCH_SIZE}" \
      "${suffix_gate_args[@]}" \
      ${DISTRO_EXTRA_ARGS} \
      --logs json \
      --write-log "${client_json}" \
      "${wandb_args[@]}" \
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
            is_loss = ("integration_test_log_marker=loss" in line) or ('"integration_test_log_marker":"loss"' in line)
            has_step = (f"step={target}" in line) or (f'"step":{target}' in line)
            if is_loss and has_step:
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

has_log_marker() {
  local path="$1"
  local marker="$2"
  "${PY}" - "$path" "$marker" <<'PY'
import sys

path = sys.argv[1]
marker = sys.argv[2]

try:
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", errors="ignore")
            if marker in line:
                raise SystemExit(0)
except FileNotFoundError:
    pass

raise SystemExit(1)
PY
}

train_last_step=$((TOTAL_STEPS - 1))
start_ts=$(date +%s)

echo "[run] Waiting for completion (train_last_step=${train_last_step}, total_steps=${TOTAL_STEPS})..."
while true; do
  now=$(date +%s)
  elapsed=$((now - start_ts))
  if (( elapsed > MAX_WAIT_SECS )); then
    echo "[run] Timed out after ${elapsed}s waiting for step ${train_last_step}." >&2
    exit 1
  fi

  # Fail fast if a process died.
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "[run] Server process exited early." >&2
    exit 1
  fi
  for pid in "${client_pids[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[run] A client process exited early (pid=${pid})." >&2
      exit 1
    fi
  done

  ok=1
  for tier in "${tiers[@]}"; do
    if ! has_loss_at_step "${LOG_ROOT}/client-tier${tier}.console.log" "${train_last_step}"; then
      ok=0
      break
    fi
  done
  if (( ok == 1 )); then
    break
  fi

  sleep 5
done

echo "[run] Reached completion; shutting down..."
if [[ "${WAIT_FOR_MODEL_EXTRACT}" == "1" ]]; then
  echo "[run] Waiting for cooldown model extract markers..."
  start_extract_ts=$(date +%s)
  while true; do
    now=$(date +%s)
    elapsed=$((now - start_extract_ts))
    if (( elapsed > WAIT_FOR_MODEL_EXTRACT_TIMEOUT_SECS )); then
      echo "[run] Timed out after ${elapsed}s waiting for cooldown extract markers." >&2
      break
    fi
    ok=1
    for tier in "${tiers[@]}"; do
      if ! has_log_marker "${LOG_ROOT}/client-tier${tier}.console.log" "Model extracted;"; then
        ok=0
        break
      fi
    done
    if (( ok == 1 )); then
      echo "[run] Cooldown extract complete."
      break
    fi
    sleep 2
  done
fi
exit 0

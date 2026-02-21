#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# This script is intended for a single 1xA100 host. It launches a small matrix of
# 2000-step runs in parallel (distinct ports + distinct metrics ports) and writes
# logs under logs/a100_parallel_2000/<timestamp>/<experiment>/.
#
# Keep the matrix small: parallel runs share one GPU and mostly contend on compute.

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
  echo "[run] WANDB_API_KEY is not set. Put it in env.local on the host." >&2
  exit 1
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

ALIGNED_BATCH_SEED="${ALIGNED_BATCH_SEED:-20260216}"
BASE_SERVER_PORT="${BASE_SERVER_PORT:-23100}"
BASE_METRICS_PORT="${BASE_METRICS_PORT:-6270}"
BASE_P2P_PORT="${BASE_P2P_PORT:-33100}"
DEVICE="${DEVICE:-cuda}"

IROH_DISCOVERY="${IROH_DISCOVERY:-local}"
IROH_RELAY="${IROH_RELAY:-disabled}"

# Raw-gradient experiment knobs (override via env).
RAW_NORM_MODE="${RAW_NORM_MODE:-match-sign-equivalent}"
RAW_SCALE_MULT="${RAW_SCALE_MULT:-0.25}"
RAW_ABS_CLIP_MULT="${RAW_ABS_CLIP_MULT:-4.0}"
GEOM_ALIGN="${GEOM_ALIGN:-1}"
GEOM_SCALE_POWER="${GEOM_SCALE_POWER:-0.5}"
RAW_ALIGN_SIGN_SCALE="${RAW_ALIGN_SIGN_SCALE:-0}"

WANDB_PROJECT="${WANDB_PROJECT:-psyche-a100}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-a100-2000-$(date -u +%Y%m%d_%H%M%S)}"

LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/a100_parallel_2000/$(date -u +%Y%m%d_%H%M%S)}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-14400}" # 4h hard cap per experiment
mkdir -p "${LOG_ROOT}"

echo "[run] log_root=${LOG_ROOT}"
echo "[run] wandb_project=${WANDB_PROJECT} wandb_group=${WANDB_GROUP}"

echo "[run] Building release binaries (server + client)..."
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

CFG_SIGN="${CFG_SIGN:-config/test-nanogpt-20m-fineweb-2000-distro-fast/state.toml}"
CFG_RAWGEOM="${CFG_RAWGEOM:-config/test-nanogpt-20m-fineweb-2000-distro-rawgeom-fast/state.toml}"

PY="${PYO3_PYTHON:-python3}"
read -r RUN_ID_SIGN TOTAL_STEPS_SIGN < <("${PY}" - "${CFG_SIGN}" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["run_id"], cfg["config"]["total_steps"])
PY
)"

read -r RUN_ID_RAWGEOM TOTAL_STEPS_RAWGEOM < <("${PY}" - "${CFG_RAWGEOM}" <<'PY'
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open(sys.argv[1], "rb") as f:
    cfg = tomllib.load(f)
print(cfg["run_id"], cfg["config"]["total_steps"])
PY
)"

wait_for_server() {
  local pid="$1"
  local port="$2"
  local retries=240

  while (( retries > 0 )); do
    if "${PY}" - <<'PY' "${port}"
import socket,sys
port=int(sys.argv[1])
s=socket.socket(); s.settimeout(0.4)
ok=s.connect_ex(("127.0.0.1", port))==0
s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[run] Server exited before becoming ready (port=${port})." >&2
      return 1
    fi

    sleep 1
    retries=$((retries - 1))
  done

  echo "[run] Timed out waiting for server on port ${port}." >&2
  return 1
}

run_one() {
  local name="$1"
  local cfg_path="$2"
  local run_id="$3"
  local total_steps="$4"
  local server_port="$5"
  local metrics_base="$6"
  local identity_base="$7"
  local p2p_base="$8"
  local env_kvs="$9"
  local client_extra="${10}"

  local run_dir="${LOG_ROOT}/${name}"
  mkdir -p "${run_dir}"

  echo "[${name}] starting (port=${server_port} metrics_base=${metrics_base})"

  (
    # Experiment-scoped env knobs (modular norm / raw scaling).
    if [[ -n "${env_kvs}" ]]; then
      # shellcheck disable=SC2086
      export ${env_kvs}
    fi

    export RUST_LOG="warn,psyche_client=info,psyche_modeling=info,psyche_centralized_server=info,psyche_centralized_client=info"

    local server_json="${run_dir}/server.jsonl"
    local server_console="${run_dir}/server.console.log"

    "${CARGO_TARGET_DIR}/release/psyche-centralized-server" run \
      --state "${cfg_path}" \
      --server-port "${server_port}" \
      --tui false \
      --logs json \
      --write-log "${server_json}" \
      --aligned-batches \
      --aligned-batches-seed "${ALIGNED_BATCH_SEED}" \
      >"${server_console}" 2>&1 &
    local server_pid=$!

    wait_for_server "${server_pid}" "${server_port}"

    # Two local clients (tier0 + tier1). Use distinct metrics ports to allow parallel experiments.
    local pids=()
    for tier in 0 1; do
      local key_int=$((identity_base + tier))
      local raw_key
      raw_key="$(printf '%064x' "${key_int}")"
      local metrics_port=$((metrics_base + tier))
      local p2p_port=$((p2p_base + tier))

      local client_json="${run_dir}/client-tier${tier}.jsonl"
      local client_console="${run_dir}/client-tier${tier}.console.log"

      RAW_IDENTITY_SECRET_KEY="${raw_key}" METRICS_LOCAL_PORT="${metrics_port}" \
      "${CARGO_TARGET_DIR}/release/psyche-centralized-client" train \
        --run-id "${run_id}" \
        --server-addr "127.0.0.1:${server_port}" \
        --device "${DEVICE}" \
        --matformer-tier "${tier}" \
        --bind-p2p-port "${p2p_port}" \
        --iroh-discovery "${IROH_DISCOVERY}" \
        --iroh-relay "${IROH_RELAY}" \
        --logs json \
        --write-log "${client_json}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
        --wandb-run "${WANDB_GROUP}-${name}-tier${tier}" \
        --matformer-helper-fraction 0 \
        --matformer-helper-rotation-interval 16 \
        ${client_extra} \
        >"${client_console}" 2>&1 &

      pids+=("$!")
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

    local train_last_step=$((total_steps - 1))

    local code=0
    local start_ts
    start_ts="$(date +%s)"
    while true; do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[${name}] server died unexpectedly; see ${server_console}" >&2
        code=1
        break
      fi
      for pid in "${pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          echo "[${name}] a client died unexpectedly; see ${run_dir}/client-tier*.console.log" >&2
          code=1
          break
        fi
      done
      if (( code != 0 )); then
        break
      fi

      if (has_loss_at_step "${run_dir}/client-tier0.jsonl" "${train_last_step}" || has_finished_at_step "${run_dir}/client-tier0.jsonl" "${total_steps}") \
        && (has_loss_at_step "${run_dir}/client-tier1.jsonl" "${train_last_step}" || has_finished_at_step "${run_dir}/client-tier1.jsonl" "${total_steps}")
      then
        echo "[${name}] reached completion; shutting down..."
        break
      fi

      local now_ts
      now_ts="$(date +%s)"
      if (( now_ts - start_ts > MAX_WAIT_SECS )); then
        echo "[${name}] timed out waiting for step ${total_steps} after ${MAX_WAIT_SECS}s; shutting down." >&2
        code=1
        break
      fi

      sleep 10
    done

    for pid in "${pids[@]}"; do
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    done

    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true

    echo "[${name}] done (exit_code=${code})"
    exit "${code}"
  )
}

# Experiment matrix.
#
# Baseline: sign/sign
# Variant:  raw/raw + raw-v2 norm + modular-geometry transport scaling (MatFormer-aware)

PIDS=()

run_one \
  "sign_baseline" \
  "${CFG_SIGN}" \
  "${RUN_ID_SIGN}" \
  "${TOTAL_STEPS_SIGN}" \
  "$((BASE_SERVER_PORT + 0))" \
  "$((BASE_METRICS_PORT + 0))" \
  11000 \
  "$((BASE_P2P_PORT + 0))" \
  "" \
  "--distro-apply-mode sign --distro-value-mode sign" \
  & PIDS+=("$!")

run_one \
  "rawgeom_v2_p05" \
  "${CFG_RAWGEOM}" \
  "${RUN_ID_RAWGEOM}" \
  "${TOTAL_STEPS_RAWGEOM}" \
  "$((BASE_SERVER_PORT + 1))" \
  "$((BASE_METRICS_PORT + 10))" \
  12000 \
  "$((BASE_P2P_PORT + 10))" \
  "PSYCHE_DISTRO_MATFORMER_GEOMETRY_ALIGN=${GEOM_ALIGN} PSYCHE_DISTRO_MATFORMER_GEOMETRY_SCALE_POWER=${GEOM_SCALE_POWER} PSYCHE_DISTRO_RAW_ALIGN_SIGN_SCALE=${RAW_ALIGN_SIGN_SCALE}" \
  "--distro-apply-mode raw --distro-value-mode raw --distro-raw-v2-enabled --distro-raw-norm-mode ${RAW_NORM_MODE} --distro-raw-scale-multiplier ${RAW_SCALE_MULT} --distro-raw-abs-clip-mult ${RAW_ABS_CLIP_MULT}" \
  & PIDS+=("$!")

echo "[run] launched ${#PIDS[@]} experiment(s)"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

echo "[run] all experiments finished (fail=${FAIL})"
exit "${FAIL}"

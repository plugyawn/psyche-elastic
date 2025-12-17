#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/psyche-centralized-spawn-clients.sh \
    --run-id <RUN_ID> \
    --server-addr <HOST:PORT> \
    --num-clients <N> \
    [--device <auto|cpu|mps|cuda...>] \
    [--tiers <comma-separated tiers>] \
    [--iroh-discovery <local|n0>] \
    [--iroh-relay <disabled|psyche|n0>] \
    [--logs <tui|console|json>] \
    [--log-dir <DIR>] \
    [--start-key <INT>] \
    [--rust-log <RUST_LOG>] \
    [-- <extra args passed to each client>]

Examples:
  scripts/psyche-centralized-spawn-clients.sh --run-id test-tiny-llama --server-addr 192.168.1.10:20000 \
    --num-clients 2 --device mps --tiers 0,2 --logs json --rust-log "warn,psyche_client=info"
EOF
}

run_id=""
server_addr=""
num_clients=""
device="auto"
tiers_csv="0"
iroh_discovery="local"
iroh_relay="disabled"
logs="json"
log_dir=""
start_key="1"
rust_log="${RUST_LOG:-warn}"

extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) run_id="${2:-}"; shift 2 ;;
    --server-addr) server_addr="${2:-}"; shift 2 ;;
    --num-clients) num_clients="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --tiers) tiers_csv="${2:-}"; shift 2 ;;
    --iroh-discovery) iroh_discovery="${2:-}"; shift 2 ;;
    --iroh-relay) iroh_relay="${2:-}"; shift 2 ;;
    --logs) logs="${2:-}"; shift 2 ;;
    --log-dir) log_dir="${2:-}"; shift 2 ;;
    --start-key) start_key="${2:-}"; shift 2 ;;
    --rust-log) rust_log="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${run_id}" || -z "${server_addr}" || -z "${num_clients}" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi

if ! [[ "${num_clients}" =~ ^[0-9]+$ ]] || [[ "${num_clients}" -le 0 ]]; then
  echo "--num-clients must be a positive integer (got: ${num_clients})." >&2
  exit 2
fi

if ! [[ "${start_key}" =~ ^[0-9]+$ ]] || [[ "${start_key}" -le 0 ]]; then
  echo "--start-key must be a positive integer (got: ${start_key})." >&2
  exit 2
fi

if [[ -z "${log_dir}" ]]; then
  ts="$(date -u +"%Y-%m-%d_%H-%M-%S")"
  log_dir="${ROOT}/logs/clients_${ts}"
fi

mkdir -p "${log_dir}"

IFS=',' read -r -a tiers <<<"${tiers_csv}"
if [[ "${#tiers[@]}" -eq 0 ]]; then
  tiers=("0")
fi

CLIENT_BIN=""
if [[ -x "${ROOT}/bin/psyche-centralized-client" ]]; then
  CLIENT_BIN="${ROOT}/bin/psyche-centralized-client"
elif [[ -x "${CARGO_TARGET_DIR}/release/psyche-centralized-client" ]]; then
  CLIENT_BIN="${CARGO_TARGET_DIR}/release/psyche-centralized-client"
elif [[ -x "${CARGO_TARGET_DIR}/debug/psyche-centralized-client" ]]; then
  CLIENT_BIN="${CARGO_TARGET_DIR}/debug/psyche-centralized-client"
else
  echo "[spawn] client binary not found; building (debug)..." >&2
  cargo build -p psyche-centralized-client
  CLIENT_BIN="${CARGO_TARGET_DIR}/debug/psyche-centralized-client"
fi

pids=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

for ((i=0; i<num_clients; i++)); do
  tier="${tiers[$((i % ${#tiers[@]}))]}"
  key_int="$((start_key + i))"
  raw_key="$(printf "%064x" "${key_int}")"
  metrics_port="$((6270 + i))"
  log_file="${log_dir}/client-${i}-tier${tier}.jsonl"

  echo "[spawn] client=${i} tier=${tier} metrics_port=${metrics_port} log=${log_file}"
  RAW_IDENTITY_SECRET_KEY="${raw_key}" METRICS_LOCAL_PORT="${metrics_port}" RUST_LOG="${rust_log}" \
    "${CLIENT_BIN}" train \
      --run-id "${run_id}" \
      --server-addr "${server_addr}" \
      --logs "${logs}" \
      --device "${device}" \
      --matformer-tier "${tier}" \
      --iroh-discovery "${iroh_discovery}" \
      --iroh-relay "${iroh_relay}" \
      --write-log "${log_file}" \
      "${extra_args[@]}" &

  pids+=("$!")
done

echo "[spawn] started ${#pids[@]} client(s); waiting..."
wait

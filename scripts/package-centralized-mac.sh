#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

OUT_DIR="${1:-${ROOT}/dist/psyche-centralized-mac}"
TAR_PATH="${2:-${ROOT}/dist/psyche-centralized-mac.tar.gz}"

mkdir -p "$(dirname "${OUT_DIR}")"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/bin" "${OUT_DIR}/scripts" "${OUT_DIR}/config"

echo "[package] building release binaries..."
cargo build --release -p psyche-centralized-server -p psyche-centralized-client

echo "[package] copying binaries..."
cp "${CARGO_TARGET_DIR}/release/psyche-centralized-server" "${OUT_DIR}/bin/"
cp "${CARGO_TARGET_DIR}/release/psyche-centralized-client" "${OUT_DIR}/bin/"

echo "[package] copying helper scripts..."
cp "${ROOT}/scripts/psyche-env.sh" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/psyche-centralized-spawn-clients.sh" "${OUT_DIR}/scripts/"
cp "${ROOT}/scripts/export_matformer_tiers.py" "${OUT_DIR}/scripts/"

cat > "${OUT_DIR}/scripts/run-server.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${HERE}/psyche-env.sh"
exec "${ROOT}/bin/psyche-centralized-server" "$@"
SH

cat > "${OUT_DIR}/scripts/run-client.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${HERE}/psyche-env.sh"
exec "${ROOT}/bin/psyche-centralized-client" "$@"
SH

chmod +x "${OUT_DIR}/scripts/run-server.sh" "${OUT_DIR}/scripts/run-client.sh" || true

echo "[package] copying configs (tiny-llama)..."
if [[ -d "${ROOT}/config/test-tiny-llama" ]]; then
  mkdir -p "${OUT_DIR}/config/test-tiny-llama"
  cp "${ROOT}/config/test-tiny-llama/state.toml" "${OUT_DIR}/config/test-tiny-llama/" || true
  cp "${ROOT}/config/test-tiny-llama/README.md" "${OUT_DIR}/config/test-tiny-llama/" || true
fi

cat > "${OUT_DIR}/README.txt" <<'TXT'
Psyche Centralized (macOS) bundle
================================

Requirements on each machine:
- Python 3 + PyTorch installed (torch must be importable; MPS works if you installed the Apple build).

Server (on one machine):
  RUST_LOG="warn,psyche_centralized_server=info" \
    bash scripts/run-server.sh run --state config/test-tiny-llama/state.toml --server-port 20000 --tui false --logs json --write-log logs/server.jsonl

Client (on each machine):
  RUST_LOG="warn,psyche_client=info,psyche_centralized_client=info" \
    bash scripts/run-client.sh train --run-id test-tiny-llama --server-addr <SERVER_LAN_IP>:20000 --logs json --write-log logs/client.jsonl --device mps --matformer-tier 2 --iroh-discovery local --iroh-relay disabled

Notes:
- `scripts/psyche-env.sh` auto-locates torch's `lib/` directory and sets DYLD_LIBRARY_PATH/LD_LIBRARY_PATH for tch-rs.
- For WAN use `--iroh-discovery n0 --iroh-relay psyche`.
- If you want smaller devices to only pull prefix (sliced) checkpoints, run
  `python scripts/export_matformer_tiers.py --src ./checkpoints/<model> --tiers 1 2`
  and point tiered clients at the generated `-tierN` directories (default load strategy is auto).
TXT

mkdir -p "$(dirname "${TAR_PATH}")"
rm -f "${TAR_PATH}"
tar -C "$(dirname "${OUT_DIR}")" -czf "${TAR_PATH}" "$(basename "${OUT_DIR}")"
echo "[package] wrote ${TAR_PATH}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "[unquarantine] Clearing Gatekeeper quarantine xattrs in: ${ROOT}"
  xattr -dr com.apple.quarantine "${ROOT}" 2>/dev/null || true
fi

chmod +x "${ROOT}/bin/"* 2>/dev/null || true
chmod +x "${ROOT}/scripts/"*.sh 2>/dev/null || true

echo "[unquarantine] Done."

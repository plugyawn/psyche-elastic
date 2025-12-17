#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/psyche-env.sh
source "${ROOT}/scripts/psyche-env.sh"

BIN="${CARGO_TARGET_DIR}/release/psyche-centralized-client"
if [[ ! -x "${BIN}" ]]; then
  cargo build --release -p psyche-centralized-client
fi

exec "${BIN}" "$@"


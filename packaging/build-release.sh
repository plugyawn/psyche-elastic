#!/bin/bash
# Build Psyche client binary for distribution
#
# This script builds the psyche-centralized-client binary with all necessary
# optimizations and prepares it for deployment.
#
# Prerequisites:
#   - Rust toolchain installed
#   - PyTorch/libtorch available (run: source scripts/psyche-env.sh)
#   - nix develop shell (recommended)
#
# Usage:
#   ./build-release.sh              # Build for current platform
#   ./build-release.sh --strip      # Build and strip debug symbols

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
STRIP_BINARY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --strip)
            STRIP_BINARY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--strip]"
            echo "  --strip    Strip debug symbols from binary"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# === Check environment ===
log_info "Checking build environment..."

# Check for libtorch
if [[ -z "${LIBTORCH:-}" && -z "${LIBTORCH_USE_PYTORCH:-}" ]]; then
    # Try to find PyTorch in venv
    if [[ -f ".venv/lib/python3.12/site-packages/torch/lib/libtorch.so" ]] || \
       [[ -f ".venv/lib/python3.12/site-packages/torch/lib/libtorch.dylib" ]]; then
        log_info "Found PyTorch in .venv, setting up environment..."
        export LIBTORCH_USE_PYTORCH=1
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:.venv/lib/python3.12/site-packages/torch/lib"
        export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:.venv/lib/python3.12/site-packages/torch/lib"
    else
        log_warn "LIBTORCH not set. You may need to run: source scripts/psyche-env.sh"
    fi
fi

# Check Rust
if ! command -v cargo &> /dev/null; then
    log_error "Cargo not found. Install Rust or enter nix develop shell."
    exit 1
fi

# === Build ===
log_info "Building psyche-centralized-client (release mode)..."
log_info "This may take several minutes..."

# Set optimization flags
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"

# Build the client binary
cargo build --release -p psyche-centralized-client 2>&1 | tail -20

if [[ ! -f "target/release/psyche-centralized-client" ]]; then
    log_error "Build failed: binary not found"
    exit 1
fi

# === Prepare output ===
mkdir -p "$SCRIPT_DIR/dist"
OUTPUT_PATH="$SCRIPT_DIR/dist/psyche-client"

cp "target/release/psyche-centralized-client" "$OUTPUT_PATH"

# Strip if requested
if [[ "$STRIP_BINARY" == "true" ]]; then
    log_info "Stripping debug symbols..."
    if command -v strip &> /dev/null; then
        strip "$OUTPUT_PATH" 2>/dev/null || log_warn "strip failed (may not be supported)"
    else
        log_warn "strip command not found, skipping"
    fi
fi

# === Report ===
BINARY_SIZE=$(ls -lh "$OUTPUT_PATH" | awk '{print $5}')
BINARY_SHA256=$(sha256sum "$OUTPUT_PATH" 2>/dev/null || shasum -a 256 "$OUTPUT_PATH" | awk '{print $1}')

cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║                      BUILD COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Output:    $OUTPUT_PATH
║  Size:      $BINARY_SIZE
║  SHA256:    ${BINARY_SHA256:0:16}...
╠══════════════════════════════════════════════════════════════════╣
║  Platform:  $(uname -s)-$(uname -m)
║  Rust:      $(rustc --version | cut -d' ' -f2)
╚══════════════════════════════════════════════════════════════════╝

Next steps:
  1. Test the binary:
     ./packaging/dist/psyche-client --help

  2. Edit install.sh with your configuration:
     - SERVER_ADDR (Tailscale IP of coordinator)
     - RUN_ID
     - BINARY_URL (or use --binary flag)

  3. Deploy to servers:
     scp packaging/install.sh packaging/dist/psyche-client user@server:/tmp/
     ssh user@server 'sudo /tmp/install.sh --tier m --binary /tmp/psyche-client'

EOF

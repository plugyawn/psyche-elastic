#!/bin/bash
# Downloads Fineweb10B dataset for Psyche training
#
# Usage:
#   ./download-fineweb.sh                    # Download to default location
#   ./download-fineweb.sh /custom/path       # Download to custom location
#
# Requires: Python 3.12 venv at /opt/psyche/venv (created by install.sh)

set -euo pipefail

# Configuration
DATA_DIR="${1:-/opt/psyche/data/fineweb10B}"
VENV_DIR="/opt/psyche/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check venv exists
if [[ ! -d "$VENV_DIR/bin" ]]; then
    log_error "Python venv not found at $VENV_DIR"
    log_error "Run install.sh first to set up the environment"
    exit 1
fi

# Print banner
cat <<EOF

===============================================================
                 FINEWEB DATASET DOWNLOADER
===============================================================
  Destination: $DATA_DIR
  Dataset:     HuggingFaceFW/fineweb (sample-10BT)
===============================================================

EOF

# Install required packages
log_info "Installing huggingface_hub and datasets..."
"$VENV_DIR/bin/pip" install --quiet --upgrade huggingface_hub datasets

# Create data directory
log_info "Creating data directory..."
mkdir -p "$DATA_DIR"

# Download dataset
log_info "Downloading Fineweb 10B sample dataset..."
log_warn "This may take a while depending on your internet connection..."

"$VENV_DIR/bin/python" -c "
import os
from datasets import load_dataset

print('Loading dataset from HuggingFace...')
ds = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT', split='train')

print(f'Dataset loaded: {len(ds):,} examples')
print(f'Saving to disk at: $DATA_DIR')

ds.save_to_disk('$DATA_DIR')
print('Done!')
"

# Verify download
if [[ -d "$DATA_DIR" ]] && [[ -f "$DATA_DIR/dataset_info.json" || -f "$DATA_DIR/state.json" ]]; then
    log_info "Dataset downloaded successfully!"
    echo ""
    echo "  Location: $DATA_DIR"
    echo "  Size: $(du -sh "$DATA_DIR" | cut -f1)"
    echo ""
else
    log_error "Download may have failed. Check $DATA_DIR for contents."
    exit 1
fi

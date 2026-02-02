#!/bin/bash
# Psyche Training Client Installer
# Usage: curl -fsSL https://your-server/install.sh | bash -s -- --tier m
#
# Tier options:
#   s/small  - MatFormer tier 2 (25% FFN width)
#   m/medium - MatFormer tier 1 (50% FFN width)
#   l/large  - MatFormer tier 0 (100% FFN width)

set -euo pipefail

# === Configuration (baked at package time) ===
# Edit these before distributing
SERVER_ADDR="${SERVER_ADDR:-100.x.x.x:20000}"  # Tailscale IP:port of coordinator
RUN_ID="${RUN_ID:-default-run}"
BINARY_URL="${BINARY_URL:-}"  # URL to download binary, or leave empty for local copy

# === Colors for output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# === Parse arguments ===
TIER="m"
BINARY_PATH=""
PACKAGE_URL=""
SKIP_DEPS=false

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --tier TIER       Set compute tier: s/small, m/medium, l/large (default: m)
    --binary PATH     Path to local psyche-client binary (instead of downloading)
    --from URL        Base URL of package server (auto-downloads binary from URL/dist/psyche-client)
    --server ADDR     Override coordinator server address (default: baked-in value)
    --run-id ID       Override run ID (default: baked-in value)
    --skip-deps       Skip system dependency installation
    -h, --help        Show this help message

Examples:
    $0 --tier s                                  # Install small tier (requires BINARY_URL set)
    $0 --tier m --from http://100.x.x.x:8080    # Download from package server
    $0 --tier l --binary ./client                # Use local binary
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --binary)
            BINARY_PATH="$2"
            shift 2
            ;;
        --from)
            PACKAGE_URL="$2"
            shift 2
            ;;
        --server)
            SERVER_ADDR="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# If --from was provided, construct BINARY_URL from it
if [[ -n "$PACKAGE_URL" ]]; then
    # Remove trailing slash if present
    PACKAGE_URL="${PACKAGE_URL%/}"
    BINARY_URL="${PACKAGE_URL}/dist/psyche-client"
fi

# === Map S/M/L to matformer tier numbers ===
case $TIER in
    s|small|S|SMALL)
        MATFORMER_TIER=2
        TIER_NAME="small"
        FFN_PERCENT="25%"
        ;;
    m|medium|M|MEDIUM)
        MATFORMER_TIER=1
        TIER_NAME="medium"
        FFN_PERCENT="50%"
        ;;
    l|large|L|LARGE)
        MATFORMER_TIER=0
        TIER_NAME="large"
        FFN_PERCENT="100%"
        ;;
    *)
        log_error "Invalid tier: $TIER (use s/m/l or small/medium/large)"
        exit 1
        ;;
esac

# === Check root ===
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# === Print banner ===
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║                    PSYCHE CLIENT INSTALLER                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Tier:           $TIER_NAME ($FFN_PERCENT FFN width)
║  MatFormer Tier: $MATFORMER_TIER
║  Server:         $SERVER_ADDR
║  Run ID:         $RUN_ID
╚══════════════════════════════════════════════════════════════════╝

EOF

# === Detect OS ===
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
else
    log_error "Cannot detect OS. Only Ubuntu/Debian supported."
    exit 1
fi

if [[ "$OS" != "ubuntu" && "$OS" != "debian" ]]; then
    log_warn "Detected OS: $OS. This script is designed for Ubuntu/Debian."
    log_warn "Proceeding anyway, but some commands may fail."
fi

# === Install system dependencies ===
if [[ "$SKIP_DEPS" == "false" ]]; then
    log_info "Installing system dependencies..."
    apt-get update -qq

    # Check for Python 3.12
    if ! command -v python3.12 &> /dev/null; then
        log_info "Installing Python 3.12..."
        # Add deadsnakes PPA for Python 3.12 if on Ubuntu
        if [[ "$OS" == "ubuntu" ]]; then
            apt-get install -y software-properties-common
            add-apt-repository -y ppa:deadsnakes/ppa
            apt-get update -qq
        fi
        apt-get install -y python3.12 python3.12-venv python3.12-dev
    else
        log_info "Python 3.12 already installed"
        apt-get install -y python3.12-venv python3.12-dev 2>/dev/null || true
    fi

    # Install other dependencies
    apt-get install -y curl ca-certificates
else
    log_info "Skipping dependency installation (--skip-deps)"
fi

# === Create install directory ===
log_info "Creating installation directory..."
mkdir -p /opt/psyche/{bin,venv,logs,data}

# === Setup Python venv with PyTorch ===
if [[ ! -d /opt/psyche/venv/bin ]]; then
    log_info "Creating Python virtual environment..."
    python3.12 -m venv /opt/psyche/venv
fi

log_info "Installing Python packages (this may take a few minutes)..."
/opt/psyche/venv/bin/pip install --quiet --upgrade pip

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    log_info "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    /opt/psyche/venv/bin/pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
else
    log_info "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    /opt/psyche/venv/bin/pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install additional dependencies
/opt/psyche/venv/bin/pip install --quiet safetensors==0.5.2

# === Get the binary ===
if [[ -n "$BINARY_PATH" ]]; then
    log_info "Copying binary from $BINARY_PATH..."
    cp "$BINARY_PATH" /opt/psyche/bin/psyche-client
elif [[ -n "$BINARY_URL" ]]; then
    log_info "Downloading binary from $BINARY_URL..."
    curl -fsSL "$BINARY_URL" -o /opt/psyche/bin/psyche-client
else
    log_error "No binary source specified. Either:"
    log_error "  - Set BINARY_URL in the script"
    log_error "  - Use --binary PATH to specify a local binary"
    exit 1
fi
chmod +x /opt/psyche/bin/psyche-client

# === Write config ===
log_info "Writing configuration..."
cat > /opt/psyche/config.env <<EOF
# Psyche Client Configuration
# Generated by install.sh on $(date -Iseconds)

# Server connection
SERVER_ADDR=${SERVER_ADDR}
RUN_ID=${RUN_ID}

# MatFormer configuration
MATFORMER_TIER=${MATFORMER_TIER}

# PyTorch library path (required for linking)
LD_LIBRARY_PATH=/opt/psyche/venv/lib/python3.12/site-packages/torch/lib
DYLD_LIBRARY_PATH=/opt/psyche/venv/lib/python3.12/site-packages/torch/lib

# Device selection (auto = CUDA if available, else CPU)
DEVICE=auto

# Uncomment to override:
# CHECKPOINT_PATH=/opt/psyche/data/checkpoint
# DATA_PATH=/opt/psyche/data/dataset
EOF

# === Install systemd service ===
log_info "Installing systemd service..."
cat > /etc/systemd/system/psyche-client.service <<EOF
[Unit]
Description=Psyche Training Client (Tier: ${TIER_NAME}, ${FFN_PERCENT} FFN)
After=network-online.target tailscaled.service
Wants=network-online.target
Documentation=https://github.com/PsycheFoundation/psyche

[Service]
Type=simple
EnvironmentFile=/opt/psyche/config.env
WorkingDirectory=/opt/psyche

ExecStart=/opt/psyche/bin/psyche-client train \\
    --server-addr \${SERVER_ADDR} \\
    --run-id \${RUN_ID} \\
    --matformer-tier \${MATFORMER_TIER} \\
    --device \${DEVICE}

# Restart policy
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Logging
StandardOutput=append:/opt/psyche/logs/client.log
StandardError=append:/opt/psyche/logs/client.log

# Resource limits (adjust as needed)
# LimitNOFILE=65535
# MemoryMax=80%

[Install]
WantedBy=multi-user.target
EOF

# === Setup log rotation ===
log_info "Configuring log rotation..."
cat > /etc/logrotate.d/psyche-client <<EOF
/opt/psyche/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# === Enable and start ===
log_info "Enabling and starting service..."
systemctl daemon-reload
systemctl enable psyche-client

# Only start if server address is configured
if [[ "$SERVER_ADDR" == "100.x.x.x:20000" ]]; then
    log_warn "Server address not configured. Edit /opt/psyche/config.env and run:"
    log_warn "  systemctl start psyche-client"
else
    systemctl start psyche-client
    sleep 2

    if systemctl is-active --quiet psyche-client; then
        log_info "Service started successfully!"
    else
        log_warn "Service may have failed to start. Check logs with:"
        log_warn "  journalctl -u psyche-client -n 50"
    fi
fi

# === Print summary ===
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║                    INSTALLATION COMPLETE                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Installation path:  /opt/psyche                                 ║
║  Configuration:      /opt/psyche/config.env                      ║
║  Logs:               /opt/psyche/logs/client.log                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Useful commands:                                                ║
║    systemctl status psyche-client    # Check service status      ║
║    systemctl restart psyche-client   # Restart after config edit ║
║    journalctl -u psyche-client -f    # Follow live logs          ║
║    tail -f /opt/psyche/logs/client.log  # View log file          ║
╚══════════════════════════════════════════════════════════════════╝

EOF

# === Verify installation ===
log_info "Installation summary:"
echo "  Binary:     $(ls -lh /opt/psyche/bin/psyche-client | awk '{print $5}')"
echo "  Venv:       $(du -sh /opt/psyche/venv 2>/dev/null | cut -f1)"
echo "  Tier:       $TIER_NAME (matformer_tier=$MATFORMER_TIER)"
echo "  Server:     $SERVER_ADDR"

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "  GPU:        None detected (CPU mode)"
fi

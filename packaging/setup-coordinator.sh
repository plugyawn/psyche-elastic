#!/bin/bash
# Sets up coordinator + tier-0 client on the L (Large) server
#
# This script:
# 1. Installs the psyche client as tier L (100% FFN)
# 2. Downloads the Fineweb dataset
# 3. Sets up the coordinator server
# 4. Creates systemd services for both coordinator and client
#
# Usage:
#   sudo ./setup-coordinator.sh --from http://package-server:8080 --checkpoint user/nanogpt-20m-psyche
#
# Prerequisites:
#   - Ubuntu/Debian Linux
#   - Root access
#   - Tailscale installed and connected (for networking)

set -euo pipefail

# === Configuration ===
PACKAGE_URL=""
HF_CHECKPOINT=""
COORDINATOR_PORT="20000"
COORDINATOR_BIND="0.0.0.0"
CONFIG_PATH="/opt/psyche/config"
RUN_ID="prod-heterogeneous-sssl"
OTEL_ENDPOINT=""
HF_TOKEN=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --from URL            Base URL of package server (required)
    --checkpoint REPO     HuggingFace checkpoint repo (required)
    --port PORT           Coordinator port (default: 20000)
    --run-id ID           Run ID (default: prod-heterogeneous-sssl)
    --otel-endpoint URL   OpenTelemetry collector endpoint (e.g., http://100.64.0.1:4318)
    --hf-token TOKEN      HuggingFace token (for private repos or rate limiting)
    --skip-data           Skip Fineweb dataset download
    -h, --help            Show this help message

Example:
    $0 --from http://100.x.x.x:8080 --checkpoint myuser/nanogpt-20m-psyche
EOF
}

SKIP_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            PACKAGE_URL="$2"
            shift 2
            ;;
        --checkpoint)
            HF_CHECKPOINT="$2"
            shift 2
            ;;
        --port)
            COORDINATOR_PORT="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --otel-endpoint)
            OTEL_ENDPOINT="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
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

# Validate required args
if [[ -z "$PACKAGE_URL" ]]; then
    log_error "--from URL is required"
    print_usage
    exit 1
fi

if [[ -z "$HF_CHECKPOINT" ]]; then
    log_error "--checkpoint REPO is required"
    print_usage
    exit 1
fi

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Get Tailscale IP for server address
TAILSCALE_IP=""
if command -v tailscale &> /dev/null; then
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -1 || echo "")
fi

if [[ -z "$TAILSCALE_IP" ]]; then
    log_warn "Tailscale IP not found. Using 0.0.0.0 for binding."
    log_warn "Clients will need to manually specify the server address."
    SERVER_ADDR="0.0.0.0:${COORDINATOR_PORT}"
else
    SERVER_ADDR="${TAILSCALE_IP}:${COORDINATOR_PORT}"
    log_info "Detected Tailscale IP: $TAILSCALE_IP"
fi

# Print banner
cat <<EOF

==============================================================================
                    PSYCHE COORDINATOR SETUP (L SERVER)
==============================================================================
  Package URL:    $PACKAGE_URL
  Checkpoint:     $HF_CHECKPOINT
  Server Address: $SERVER_ADDR
  Run ID:         $RUN_ID
  OTEL Endpoint:  ${OTEL_ENDPOINT:-<not set>}
==============================================================================

EOF

# === Step 1: Install psyche client as tier L ===
log_info "Step 1: Installing Psyche client (tier L)..."

# Build install.sh arguments
INSTALL_ARGS=(
    --tier l
    --from "$PACKAGE_URL"
    --checkpoint "$HF_CHECKPOINT"
    --server "$SERVER_ADDR"
    --run-id "$RUN_ID"
)
if [[ -n "$OTEL_ENDPOINT" ]]; then
    INSTALL_ARGS+=(--otel-endpoint "$OTEL_ENDPOINT")
fi
if [[ -n "$HF_TOKEN" ]]; then
    INSTALL_ARGS+=(--hf-token "$HF_TOKEN")
fi

curl -fsSL "${PACKAGE_URL}/install.sh" | bash -s -- "${INSTALL_ARGS[@]}"

# === Step 2: Download Fineweb dataset ===
if [[ "$SKIP_DATA" == "false" ]]; then
    log_info "Step 2: Downloading Fineweb dataset..."
    curl -fsSL "${PACKAGE_URL}/download-fineweb.sh" | bash
else
    log_info "Step 2: Skipping Fineweb dataset download (--skip-data)"
fi

# === Step 3: Download coordinator config ===
log_info "Step 3: Setting up coordinator configuration..."
mkdir -p "$CONFIG_PATH"

# Download the state.toml from package server
curl -fsSL "${PACKAGE_URL}/config/production-heterogeneous/state.toml" \
    -o "$CONFIG_PATH/state.toml"

# Update the checkpoint repo in the config
sed -i "s|PLACEHOLDER/nanogpt-20m-psyche|${HF_CHECKPOINT}|g" "$CONFIG_PATH/state.toml"

log_info "Coordinator config saved to $CONFIG_PATH/state.toml"

# === Step 4: Download and install coordinator binary ===
log_info "Step 4: Installing coordinator binary..."
curl -fsSL "${PACKAGE_URL}/dist/psyche-server" -o /opt/psyche/bin/psyche-server
chmod +x /opt/psyche/bin/psyche-server

# === Step 5: Create coordinator systemd service ===
log_info "Step 5: Creating coordinator systemd service..."
cat > /etc/systemd/system/psyche-coordinator.service <<EOF
[Unit]
Description=Psyche Training Coordinator
After=network-online.target tailscaled.service
Wants=network-online.target
Documentation=https://github.com/PsycheFoundation/psyche

[Service]
Type=simple
WorkingDirectory=/opt/psyche
EnvironmentFile=/opt/psyche/config.env

ExecStart=/opt/psyche/bin/psyche-server \\
    --bind ${COORDINATOR_BIND}:${COORDINATOR_PORT} \\
    --config-path ${CONFIG_PATH}

# Restart policy
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Logging
StandardOutput=append:/opt/psyche/logs/coordinator.log
StandardError=append:/opt/psyche/logs/coordinator.log

[Install]
WantedBy=multi-user.target
EOF

# === Step 6: Setup log rotation for coordinator ===
log_info "Step 6: Configuring log rotation..."
cat > /etc/logrotate.d/psyche-coordinator <<EOF
/opt/psyche/logs/coordinator.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# === Step 7: Enable services (but don't start yet) ===
log_info "Step 7: Enabling services..."
systemctl daemon-reload
systemctl enable psyche-coordinator
systemctl enable psyche-client

# Print summary
cat <<EOF

==============================================================================
                    COORDINATOR SETUP COMPLETE
==============================================================================

  Coordinator Config: $CONFIG_PATH/state.toml
  Coordinator Logs:   /opt/psyche/logs/coordinator.log
  Client Config:      /opt/psyche/config.env
  Client Logs:        /opt/psyche/logs/client.log

  Server Address:     $SERVER_ADDR

------------------------------------------------------------------------------
  STARTUP SEQUENCE:
------------------------------------------------------------------------------

  1. Start the coordinator first:
     sudo systemctl start psyche-coordinator

  2. Wait for coordinator to be ready, then start the local client:
     sudo systemctl start psyche-client

  3. Start clients on other S servers (they should connect automatically)

  4. Monitor progress:
     journalctl -u psyche-coordinator -f   # Coordinator logs
     journalctl -u psyche-client -f        # Client logs

------------------------------------------------------------------------------
  COMMANDS FOR S SERVERS:
------------------------------------------------------------------------------

  curl -fsSL ${PACKAGE_URL}/install.sh | sudo bash -s -- \\
    --tier s --from ${PACKAGE_URL} \\
    --checkpoint ${HF_CHECKPOINT}-tier2 \\
    --server ${SERVER_ADDR}${OTEL_ENDPOINT:+ \\
    --otel-endpoint ${OTEL_ENDPOINT}}${HF_TOKEN:+ \\
    --hf-token ${HF_TOKEN}}

  # Then download Fineweb on each S server:
  curl -fsSL ${PACKAGE_URL}/download-fineweb.sh | sudo bash

==============================================================================

EOF

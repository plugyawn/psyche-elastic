#!/bin/bash
# Psyche Training Client Uninstaller
# Removes all installed components and configuration

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[ERROR]${NC} This script must be run as root (use sudo)"
    exit 1
fi

# Parse arguments
KEEP_LOGS=false
KEEP_DATA=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-logs)
            KEEP_LOGS=true
            shift
            ;;
        --keep-data)
            KEEP_DATA=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --keep-logs   Don't delete /opt/psyche/logs"
            echo "  --keep-data   Don't delete /opt/psyche/data"
            echo "  --force, -f   Don't prompt for confirmation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Confirmation
if [[ "$FORCE" == "false" ]]; then
    echo "This will remove the Psyche training client from this system."
    echo ""
    echo "The following will be deleted:"
    echo "  - /opt/psyche/bin"
    echo "  - /opt/psyche/venv"
    [[ "$KEEP_LOGS" == "false" ]] && echo "  - /opt/psyche/logs"
    [[ "$KEEP_DATA" == "false" ]] && echo "  - /opt/psyche/data"
    echo "  - /etc/systemd/system/psyche-client.service"
    echo "  - /etc/logrotate.d/psyche-client"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Stop and disable service
log_info "Stopping and disabling psyche-client service..."
systemctl stop psyche-client 2>/dev/null || true
systemctl disable psyche-client 2>/dev/null || true

# Remove systemd unit
if [[ -f /etc/systemd/system/psyche-client.service ]]; then
    rm -f /etc/systemd/system/psyche-client.service
    log_info "Removed systemd service"
fi

systemctl daemon-reload

# Remove logrotate config
if [[ -f /etc/logrotate.d/psyche-client ]]; then
    rm -f /etc/logrotate.d/psyche-client
    log_info "Removed logrotate configuration"
fi

# Remove installation directory
if [[ -d /opt/psyche ]]; then
    # Always remove bin and venv
    rm -rf /opt/psyche/bin
    rm -rf /opt/psyche/venv
    rm -f /opt/psyche/config.env
    log_info "Removed binaries and venv"

    # Conditionally remove logs
    if [[ "$KEEP_LOGS" == "false" && -d /opt/psyche/logs ]]; then
        rm -rf /opt/psyche/logs
        log_info "Removed logs"
    elif [[ "$KEEP_LOGS" == "true" && -d /opt/psyche/logs ]]; then
        log_warn "Keeping /opt/psyche/logs"
    fi

    # Conditionally remove data
    if [[ "$KEEP_DATA" == "false" && -d /opt/psyche/data ]]; then
        rm -rf /opt/psyche/data
        log_info "Removed data"
    elif [[ "$KEEP_DATA" == "true" && -d /opt/psyche/data ]]; then
        log_warn "Keeping /opt/psyche/data"
    fi

    # Remove directory if empty
    rmdir /opt/psyche 2>/dev/null || log_warn "/opt/psyche not empty, keeping directory"
fi

cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║                  UNINSTALL COMPLETE                              ║
╚══════════════════════════════════════════════════════════════════╝

Psyche training client has been removed from this system.

EOF

if [[ "$KEEP_LOGS" == "true" || "$KEEP_DATA" == "true" ]]; then
    echo "Note: Some directories were kept as requested:"
    [[ "$KEEP_LOGS" == "true" ]] && echo "  - /opt/psyche/logs"
    [[ "$KEEP_DATA" == "true" ]] && echo "  - /opt/psyche/data"
    echo ""
fi

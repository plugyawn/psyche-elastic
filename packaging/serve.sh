#!/bin/bash
# Serve Psyche packaging files over HTTP for easy deployment
#
# Usage:
#   ./serve.sh           # Serve on port 8080
#   ./serve.sh 9000      # Serve on custom port
#
# Then on target servers:
#   curl -fsSL http://<ip>:8080/install.sh | sudo bash -s -- --tier m

set -euo pipefail

PORT="${1:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get IP addresses
get_tailscale_ip() {
    if command -v tailscale &> /dev/null; then
        tailscale ip -4 2>/dev/null | head -1
    fi
}

get_local_ip() {
    if [[ "$(uname)" == "Darwin" ]]; then
        ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo ""
    else
        hostname -I 2>/dev/null | awk '{print $1}' || echo ""
    fi
}

TAILSCALE_IP=$(get_tailscale_ip)
LOCAL_IP=$(get_local_ip)

# Check if binary exists
if [[ ! -f "$SCRIPT_DIR/dist/psyche-client" ]]; then
    echo -e "${YELLOW}[WARN]${NC} Binary not found at dist/psyche-client"
    echo "       Run ./build-release.sh first, or clients will need --binary flag"
    echo ""
fi

# Print banner
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║                   PSYCHE PACKAGE SERVER                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Serving: $SCRIPT_DIR
║  Port:    $PORT
╠══════════════════════════════════════════════════════════════════╣
EOF

if [[ -n "$TAILSCALE_IP" ]]; then
    echo -e "║  ${GREEN}Tailscale:${NC} http://$TAILSCALE_IP:$PORT"
fi
if [[ -n "$LOCAL_IP" ]]; then
    echo -e "║  ${CYAN}Local:${NC}     http://$LOCAL_IP:$PORT"
fi
echo "║  Localhost: http://127.0.0.1:$PORT"

cat <<EOF
╚══════════════════════════════════════════════════════════════════╝

EOF

# Print install commands
echo -e "${GREEN}Install commands for target servers:${NC}"
echo ""

BINARY_READY=""
if [[ -f "$SCRIPT_DIR/dist/psyche-client" ]]; then
    BINARY_READY="yes"
fi

if [[ -n "$TAILSCALE_IP" ]]; then
    echo -e "  ${CYAN}# Via Tailscale (recommended):${NC}"
    if [[ -n "$BINARY_READY" ]]; then
        echo "  curl -fsSL http://$TAILSCALE_IP:$PORT/install.sh | sudo bash -s -- --tier m --from http://$TAILSCALE_IP:$PORT"
    else
        echo "  curl -fsSL http://$TAILSCALE_IP:$PORT/install.sh -o /tmp/install.sh"
        echo "  # Copy binary separately, then:"
        echo "  sudo bash /tmp/install.sh --tier m --binary /path/to/psyche-client"
    fi
    echo ""
fi

if [[ -n "$LOCAL_IP" ]]; then
    echo -e "  ${CYAN}# Via local network:${NC}"
    if [[ -n "$BINARY_READY" ]]; then
        echo "  curl -fsSL http://$LOCAL_IP:$PORT/install.sh | sudo bash -s -- --tier m --from http://$LOCAL_IP:$PORT"
    else
        echo "  curl -fsSL http://$LOCAL_IP:$PORT/install.sh -o /tmp/install.sh"
        echo "  sudo bash /tmp/install.sh --tier m --binary /path/to/psyche-client"
    fi
    echo ""
fi

echo -e "${CYAN}Tier options:${NC}"
echo "  --tier s   Small  (25% FFN, low GPU memory)"
echo "  --tier m   Medium (50% FFN, moderate GPU memory)"
echo "  --tier l   Large  (100% FFN, full GPU memory)"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start HTTP server
cd "$SCRIPT_DIR"

if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT" --bind 0.0.0.0
elif command -v python &> /dev/null; then
    python -m http.server "$PORT" --bind 0.0.0.0
else
    echo "Error: Python not found. Install Python 3 to use this script."
    exit 1
fi

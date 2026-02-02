# Psyche Server Deployment Packaging

One-command installer for deploying Psyche training clients to fresh Linux servers.

## Quick Start

### On Your Build Machine

1. Build the client binary:
   ```bash
   ./build-release.sh
   ```

2. Edit `install.sh` with your configuration:
   ```bash
   SERVER_ADDR="100.x.x.x:20000"  # Your Tailscale server IP:port
   RUN_ID="production-run"        # Your run ID
   BINARY_URL="https://..."       # Where to download binary (optional)
   ```

### On Each Target Server

```bash
# Option 1: One-liner with package server (recommended)
# Start serve.sh on your build machine, then on target:
curl -fsSL http://100.x.x.x:8080/install.sh | sudo bash -s -- --tier m --from http://100.x.x.x:8080

# Option 2: Copy files and run locally
scp install.sh dist/psyche-client user@server:/tmp/
ssh user@server 'sudo /tmp/install.sh --tier m --binary /tmp/psyche-client'
```

## Tier Options

| Tier | CLI Flag | MatFormer Tier | FFN Width | GPU Memory |
|------|----------|----------------|-----------|------------|
| Small | `--tier s` | 2 | 25% | Low |
| Medium | `--tier m` | 1 | 50% | Medium |
| Large | `--tier l` | 0 | 100% | High |

## Files

| File | Purpose |
|------|---------|
| `install.sh` | Main installer script |
| `build-release.sh` | Builds client binary for distribution |
| `serve.sh` | HTTP server for distributing packages over network |
| `uninstall.sh` | Removes installation |
| `dist/` | Output directory for built binaries |

## Installation Details

### What Gets Installed

```
/opt/psyche/
├── bin/
│   └── psyche-client              # Pre-built binary
├── venv/                          # Python 3.12 + PyTorch
├── config.env                     # Runtime configuration
├── data/                          # Checkpoints and datasets
└── logs/                          # Runtime logs

/etc/systemd/system/
└── psyche-client.service          # Systemd unit file

/etc/logrotate.d/
└── psyche-client                  # Log rotation config
```

### System Requirements

- Ubuntu 20.04+ or Debian 11+
- Python 3.12 (auto-installed if missing)
- NVIDIA GPU with CUDA (optional, falls back to CPU)
- Tailscale configured (for server connectivity)

## Usage

### Install Command

```bash
sudo ./install.sh [OPTIONS]

Options:
    --tier TIER       Set compute tier: s/small, m/medium, l/large (default: m)
    --binary PATH     Path to local psyche-client binary
    --from URL        Base URL of package server (downloads binary from URL/dist/psyche-client)
    --server ADDR     Override coordinator server address
    --run-id ID       Override run ID
    --skip-deps       Skip system dependency installation
    -h, --help        Show help message
```

### Serve Command

```bash
./serve.sh [PORT]    # Default port: 8080
```

Starts an HTTP server that hosts the packaging files. Shows copy-paste commands for each detected network interface (Tailscale, local network).

### Post-Installation

```bash
# Check service status
systemctl status psyche-client

# View logs
journalctl -u psyche-client -f
tail -f /opt/psyche/logs/client.log

# Restart after config changes
systemctl restart psyche-client

# Edit configuration
nano /opt/psyche/config.env
```

### Uninstall

```bash
sudo ./uninstall.sh [OPTIONS]

Options:
    --keep-logs   Don't delete log files
    --keep-data   Don't delete data directory
    --force, -f   Skip confirmation prompt
```

## Configuration

### Server Configuration (`/opt/psyche/config.env`)

```bash
# Server connection
SERVER_ADDR=100.x.x.x:20000
RUN_ID=default-run

# MatFormer configuration
MATFORMER_TIER=1

# PyTorch library path
LD_LIBRARY_PATH=/opt/psyche/venv/lib/python3.12/site-packages/torch/lib

# Device selection
DEVICE=auto

# Optional overrides
# CHECKPOINT_PATH=/opt/psyche/data/checkpoint
# DATA_PATH=/opt/psyche/data/dataset
```

### Baking Configuration

Before distributing, edit `install.sh` to set default values:

```bash
SERVER_ADDR="100.x.x.x:20000"  # Your Tailscale coordinator IP
RUN_ID="production-run"         # Default run ID
BINARY_URL="https://..."        # Binary download URL (or leave empty)
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
journalctl -u psyche-client -n 100 --no-pager

# Verify binary works
/opt/psyche/bin/psyche-client --help

# Check libtorch linking
LD_LIBRARY_PATH=/opt/psyche/venv/lib/python3.12/site-packages/torch/lib \
  ldd /opt/psyche/bin/psyche-client
```

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
/opt/psyche/venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

### Connection Issues

```bash
# Verify Tailscale is connected
tailscale status

# Test connectivity to server
nc -zv <SERVER_IP> <SERVER_PORT>
```

## Air-Gapped Installation

For servers without internet access:

1. On a connected machine:
   ```bash
   ./build-release.sh
   # Also download Python packages for offline install
   /opt/psyche/venv/bin/pip download -d ./packages torch==2.5.1 safetensors==0.5.2
   ```

2. Transfer all files to target server

3. Install offline:
   ```bash
   sudo ./install.sh --tier m --binary ./psyche-client --skip-deps
   # Manually install Python packages from ./packages/
   ```

## Security Notes

- The installer requires root access
- Configuration files are readable only by root
- Consider using Tailscale ACLs to restrict client connectivity
- Don't commit real server addresses to version control

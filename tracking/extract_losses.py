#!/usr/bin/env python3
"""Extract tier-0 losses from log and write to JSON."""
import json
import re
import sys
import time

# ANSI escape code pattern
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')

def extract_losses(log_path):
    """Extract tier-0 losses from log file."""
    losses = {}
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Strip ANSI codes
                line = ANSI_ESCAPE.sub('', line)

                # Match: step=N matformer_tier=0 ... loss=X.XXX
                if 'matformer_tier=0' in line and 'loss=' in line:
                    step_match = re.search(r'step=(\d+)', line)
                    loss_match = re.search(r'loss=([0-9.]+)', line)
                    if step_match and loss_match:
                        step = int(step_match.group(1))
                        loss = float(loss_match.group(1))
                        # Take first tier-0 loss per step (or average if you prefer)
                        if step not in losses:
                            losses[step] = loss
    except FileNotFoundError:
        pass
    return losses

def main():
    if len(sys.argv) < 3:
        print("Usage: extract_losses.py <log_path> <output_json> [--watch]")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2]
    watch = len(sys.argv) > 3 and sys.argv[3] == '--watch'

    while True:
        losses = extract_losses(log_path)
        with open(output_path, 'w') as f:
            json.dump(losses, f)

        if not watch:
            break
        time.sleep(2)

if __name__ == '__main__':
    main()

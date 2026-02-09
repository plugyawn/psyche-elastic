#!/usr/bin/env python3
"""Small Lambda Cloud helper for launching and managing A100 instances."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_API_BASE = "https://cloud.lambda.ai/api/v1"


def _load_env_local(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            out[key] = val
    return out


def _resolve_api_key(explicit_key: str | None, env_file: Path) -> str:
    if explicit_key:
        return explicit_key
    from_env = os.environ.get("LAMBDA_API_KEY")
    if from_env:
        return from_env
    local = _load_env_local(env_file).get("LAMBDA_API_KEY")
    if local:
        return local
    raise SystemExit(
        "Missing Lambda API key. Set LAMBDA_API_KEY or add it to env.local."
    )


def _api_request(
    api_base: str, api_key: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{path}"
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method=method,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "psyche-lambda-helper/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if not data:
                return {}
            return json.loads(data)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"error": raw}
        raise SystemExit(f"Lambda API error {e.code} on {path}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Network error on {path}: {e}") from e


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _cmd_instance_types(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    resp = _api_request(api_base, api_key, "GET", "/instance-types")
    raw_items = resp.get("data", [])
    if isinstance(raw_items, dict):
        items = list(raw_items.values())
    elif isinstance(raw_items, list):
        items = raw_items
    else:
        raise SystemExit(f"Unexpected /instance-types payload type: {type(raw_items).__name__}")

    rows: list[list[str]] = []
    needle = (args.contains or "").lower()
    for item in items:
        inst = item.get("instance_type", {})
        name = str(inst.get("name", ""))
        if needle and needle not in name.lower():
            continue
        regions = item.get("regions_with_capacity_available", [])
        region_names = [str(r.get("name", "")) for r in regions]
        if args.available_only and not region_names:
            continue
        price = inst.get("price_cents_per_hour")
        price_str = f"${(price or 0) / 100:.2f}/hr" if price is not None else "n/a"
        rows.append(
            [
                name,
                price_str,
                ",".join(region_names) if region_names else "-",
                str(len(region_names)),
            ]
        )

    rows.sort(key=lambda r: r[0])
    if args.json:
        print(json.dumps(items, indent=2))
        return
    _print_table(rows, ["instance_type", "price", "regions", "region_count"])


def _cmd_instances(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    resp = _api_request(api_base, api_key, "GET", "/instances")
    items = resp.get("data", [])
    if args.json:
        print(json.dumps(items, indent=2))
        return

    rows: list[list[str]] = []
    for inst in items:
        rows.append(
            [
                str(inst.get("id", "")),
                str(inst.get("name", "")),
                str(inst.get("status", "")),
                str(inst.get("ip", "")),
                str((inst.get("region") or {}).get("name", "")),
                str((inst.get("instance_type") or {}).get("name", "")),
            ]
        )
    _print_table(rows, ["id", "name", "status", "ip", "region", "type"])


def _cmd_ssh_keys(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    resp = _api_request(api_base, api_key, "GET", "/ssh-keys")
    items = resp.get("data", [])
    if args.json:
        print(json.dumps(items, indent=2))
        return

    rows = [[str(k.get("id", "")), str(k.get("name", ""))] for k in items]
    _print_table(rows, ["id", "name"])


def _lookup_instance(instance_id: str, api_base: str, api_key: str) -> dict[str, Any]:
    resp = _api_request(api_base, api_key, "GET", f"/instances/{instance_id}")
    return resp.get("data", {})


def _cmd_launch(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    payload: dict[str, Any] = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [args.ssh_key],
    }
    if args.name:
        payload["name"] = args.name
    if args.hostname:
        payload["hostname"] = args.hostname

    resp = _api_request(api_base, api_key, "POST", "/instance-operations/launch", payload)
    ids = resp.get("data", {}).get("instance_ids", [])
    if not ids:
        raise SystemExit(f"Launch returned no instance IDs: {resp}")

    print("launched_instance_ids:", ",".join(ids))
    for iid in ids:
        print(f"instance_id={iid}")

    if not args.wait_active:
        return

    deadline = time.time() + args.wait_timeout
    for iid in ids:
        while True:
            inst = _lookup_instance(iid, api_base, api_key)
            status = str(inst.get("status", "unknown"))
            ip = str(inst.get("ip", ""))
            print(f"[wait] {iid} status={status} ip={ip}")
            if status == "active":
                user = args.ssh_user
                if ip:
                    print(f"ssh_cmd=ssh {user}@{ip}")
                break
            if status in {"terminated", "terminating", "preempted", "unhealthy"}:
                raise SystemExit(f"Instance {iid} entered terminal state: {status}")
            if time.time() > deadline:
                raise SystemExit(f"Timed out waiting for active status for instance {iid}.")
            time.sleep(args.poll_interval)


def _cmd_terminate(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    ids = [part.strip() for part in args.instance_ids.split(",") if part.strip()]
    if not ids:
        raise SystemExit("No instance IDs supplied.")
    payload = {"instance_ids": ids}
    _api_request(api_base, api_key, "POST", "/instance-operations/terminate", payload)
    print("terminated:", ",".join(ids))


def _cmd_ssh(args: argparse.Namespace, api_base: str, api_key: str) -> None:
    inst = _lookup_instance(args.instance_id, api_base, api_key)
    status = str(inst.get("status", "unknown"))
    ip = str(inst.get("ip", ""))
    if not ip:
        raise SystemExit(f"Instance {args.instance_id} has no public IP yet (status={status}).")
    print(f"status={status}")
    print(f"ssh {args.user}@{ip}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lambda Cloud helper for Psyche A100 runs.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--env-file", default="env.local")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_types = sub.add_parser("instance-types", help="List instance types")
    p_types.add_argument("--contains", default="a100", help="Name substring filter")
    p_types.add_argument("--available-only", action="store_true", help="Keep only available types")
    p_types.add_argument("--json", action="store_true")

    p_instances = sub.add_parser("instances", help="List instances")
    p_instances.add_argument("--json", action="store_true")

    p_keys = sub.add_parser("ssh-keys", help="List SSH keys")
    p_keys.add_argument("--json", action="store_true")

    p_launch = sub.add_parser("launch", help="Launch an instance")
    p_launch.add_argument("--region", required=True)
    p_launch.add_argument("--instance-type", required=True)
    p_launch.add_argument("--ssh-key", required=True)
    p_launch.add_argument("--name")
    p_launch.add_argument("--hostname")
    p_launch.add_argument("--wait-active", action="store_true")
    p_launch.add_argument("--wait-timeout", type=int, default=900)
    p_launch.add_argument("--poll-interval", type=int, default=10)
    p_launch.add_argument("--ssh-user", default="ubuntu")

    p_term = sub.add_parser("terminate", help="Terminate instance(s)")
    p_term.add_argument("--instance-ids", required=True, help="Comma-separated instance IDs")

    p_ssh = sub.add_parser("ssh", help="Print SSH command for an instance")
    p_ssh.add_argument("--instance-id", required=True)
    p_ssh.add_argument("--user", default="ubuntu")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    api_key = _resolve_api_key(args.api_key, Path(args.env_file))

    if args.cmd == "instance-types":
        _cmd_instance_types(args, args.api_base, api_key)
        return
    if args.cmd == "instances":
        _cmd_instances(args, args.api_base, api_key)
        return
    if args.cmd == "ssh-keys":
        _cmd_ssh_keys(args, args.api_base, api_key)
        return
    if args.cmd == "launch":
        _cmd_launch(args, args.api_base, api_key)
        return
    if args.cmd == "terminate":
        _cmd_terminate(args, args.api_base, api_key)
        return
    if args.cmd == "ssh":
        _cmd_ssh(args, args.api_base, api_key)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

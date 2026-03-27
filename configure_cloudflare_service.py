#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, parse, request


API_BASE = "https://api.cloudflare.com/client/v4"


class CloudflareError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a named Cloudflare tunnel and optionally persist its URL/token into an env file."
    )
    parser.add_argument("--api-token", required=True, help="Cloudflare API token")
    parser.add_argument("--account-id", required=True, help="Cloudflare account ID")
    parser.add_argument("--zone-id", required=True, help="Cloudflare zone ID")
    parser.add_argument("--zone-name", required=True, help="Cloudflare zone name, for example yall.uz")
    parser.add_argument("--tunnel-name", required=True, help="Named tunnel to create or reuse")
    parser.add_argument("--hostname", required=True, help="Hostname label or FQDN to publish")
    parser.add_argument("--service", required=True, help="Origin service, for example http://127.0.0.1:8000")
    parser.add_argument("--env-file", help="Optional env file to update")
    parser.add_argument(
        "--token-env-key",
        default="CLOUDFLARE_TUNNEL_TOKEN",
        help="Env key used when writing the tunnel token",
    )
    parser.add_argument(
        "--url-env-key",
        default="PUBLIC_API_BASE_URL",
        help="Env key used when writing the public HTTPS URL",
    )
    return parser.parse_args()


def cf_request(
    api_token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    if query:
        url = f"{url}?{parse.urlencode(query)}"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=data, headers=headers, method=method)

    try:
        with request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CloudflareError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise CloudflareError(f"{method} {path} failed: {exc}") from exc

    if not body.get("success"):
        raise CloudflareError(f"{method} {path} failed: {json.dumps(body.get('errors', []))}")

    return body


def get_or_create_tunnel(api_token: str, account_id: str, tunnel_name: str) -> dict[str, Any]:
    tunnels = cf_request(
        api_token,
        "GET",
        f"/accounts/{account_id}/cfd_tunnel",
        query={"name": tunnel_name},
    ).get("result", [])
    if tunnels:
        return tunnels[0]

    return cf_request(
        api_token,
        "POST",
        f"/accounts/{account_id}/cfd_tunnel",
        payload={
            "name": tunnel_name,
            "config_src": "cloudflare",
            "tunnel_secret": base64.b64encode(os.urandom(32)).decode("ascii"),
        },
    )["result"]


def get_tunnel_token(api_token: str, account_id: str, tunnel_id: str) -> str:
    token = cf_request(
        api_token,
        "GET",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token",
    ).get("result")
    if not token:
        raise CloudflareError("Cloudflare did not return a tunnel token")
    return token


def put_tunnel_config(api_token: str, account_id: str, tunnel_id: str, hostname: str, service: str) -> None:
    cf_request(
        api_token,
        "PUT",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations",
        payload={
            "config": {
                "ingress": [
                    {"hostname": hostname, "service": service},
                    {"service": "http_status:404"},
                ]
            }
        },
    )


def get_dns_record(api_token: str, zone_id: str, hostname: str) -> dict[str, Any] | None:
    records = cf_request(
        api_token,
        "GET",
        f"/zones/{zone_id}/dns_records",
        query={"name": hostname},
    ).get("result", [])
    return records[0] if records else None


def upsert_dns_cname(api_token: str, zone_id: str, hostname: str, target: str) -> None:
    payload = {
        "type": "CNAME",
        "name": hostname,
        "content": target,
        "ttl": 1,
        "proxied": True,
    }
    existing = get_dns_record(api_token, zone_id, hostname)
    if existing:
        cf_request(api_token, "PUT", f"/zones/{zone_id}/dns_records/{existing['id']}", payload=payload)
    else:
        cf_request(api_token, "POST", f"/zones/{zone_id}/dns_records", payload=payload)


def update_env_file(env_file: str, updates: dict[str, str]) -> None:
    path = Path(env_file)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output: list[str] = []
    seen: set[str] = set()

    for line in existing_lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            output.append(line)
            continue

        key, _ = line.split("=", 1)
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    hostname = args.hostname if "." in args.hostname else f"{args.hostname}.{args.zone_name}"

    try:
        tunnel = get_or_create_tunnel(args.api_token, args.account_id, args.tunnel_name)
        tunnel_id = tunnel["id"]
        tunnel_token = get_tunnel_token(args.api_token, args.account_id, tunnel_id)
        put_tunnel_config(args.api_token, args.account_id, tunnel_id, hostname, args.service)
        upsert_dns_cname(args.api_token, args.zone_id, hostname, f"{tunnel_id}.cfargotunnel.com")
    except CloudflareError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.env_file:
        update_env_file(
            args.env_file,
            {
                args.token_env_key: tunnel_token,
                args.url_env_key: f"https://{hostname}",
            },
        )

    print(f"Tunnel ID: {tunnel_id}")
    print(f"Public URL: https://{hostname}")
    print("Tunnel token written to output only as <redacted>.")
    if args.env_file:
        print(f"Updated env file: {args.env_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
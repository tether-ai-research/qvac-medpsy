"""Start ``vllm serve`` in the background, wait until it answers, then return.

Takes the same arguments as ``vllm serve``; the server keeps running after this command exits
and is stopped with the printed ``kill`` command::

    medpsy-eval serve Qwen/QwQ-32B --revision <sha> --port 8001 \\
        --reasoning-parser deepseek_r1 --max-model-len 32768 --tensor-parallel-size 2
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence

import httpx


def build_command(model: str, host: str, port: int, extra: Sequence[str]) -> list[str]:
    return ["vllm", "serve", model, "--host", host, "--port", str(port), *extra]


def wait_ready(base_url: str, timeout: float, proc: subprocess.Popen | None = None) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vllm exited with code {proc.returncode} before becoming ready")
        try:
            r = httpx.get(f"{base_url}/models", timeout=5)
            if r.status_code == 200:
                return [m["id"] for m in r.json().get("data", [])]
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"no server at {base_url} after {timeout:.0f}s")


def stop(proc: subprocess.Popen, grace: float = 60) -> None:
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Our options, and everything else in order for ``vllm serve``."""
    ap = argparse.ArgumentParser(
        prog="medpsy-eval serve", allow_abbrev=False,
        description="Start `vllm serve` with the same arguments in the background and return "
                    "once it answers.")
    ap.add_argument("model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-log", help="vLLM output file (default: vllm-<port>.log)")
    ap.add_argument("--ready-timeout", type=float, default=1800,
                    help="seconds to wait for the server (default: 1800)")
    return ap.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args, extra = parse_args(sys.argv[1:] if argv is None else argv)
    log_path = args.server_log or f"vllm-{args.port}.log"
    with open(log_path, "a") as log:
        proc = subprocess.Popen(build_command(args.model, args.host, args.port, extra),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base_url = f"http://{args.host}:{args.port}/v1"
    try:
        wait_ready(base_url, args.ready_timeout, proc)
    except (RuntimeError, TimeoutError) as e:
        stop(proc)
        raise SystemExit(f"medpsy-eval serve: {e}; see {log_path}") from None
    print(f"ready: {base_url} (log: {log_path}; stop with: kill -- -{proc.pid})", flush=True)


if __name__ == "__main__":
    main()

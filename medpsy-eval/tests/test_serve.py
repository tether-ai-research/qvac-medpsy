"""``medpsy-eval serve``: vLLM arguments pass through; the server outlives the command."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time

import httpx
import pytest

from medpsy_eval import serve


def test_vllm_arguments_pass_through_in_order() -> None:
    args, extra = serve.parse_args([
        "Qwen/QwQ-32B", "--revision", "abc", "--port", "8001", "--reasoning-parser",
        "deepseek_r1", "--server-log", "judge.log", "--tensor-parallel-size", "2"])

    assert (args.model, args.port, args.server_log) == ("Qwen/QwQ-32B", 8001, "judge.log")
    assert serve.build_command(args.model, args.host, args.port, extra) == [
        "vllm", "serve", "Qwen/QwQ-32B", "--host", "127.0.0.1", "--port", "8001",
        "--revision", "abc", "--reasoning-parser", "deepseek_r1", "--tensor-parallel-size", "2"]


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    """A ``vllm`` on PATH that answers ``/v1/models`` like a vLLM server."""
    models = tmp_path / "www" / "v1" / "models"
    models.parent.mkdir(parents=True)
    models.write_text(json.dumps({"data": [{"id": "served"}]}))
    script = tmp_path / "bin" / "vllm"
    script.parent.mkdir()
    script.write_text(f'#!/bin/sh\n# vllm serve MODEL --host HOST --port PORT\n'
                      f'exec {sys.executable} -m http.server "$6" --bind "$4" '
                      f'--directory {models.parents[1]}\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{script.parent}{os.pathsep}{os.environ['PATH']}")
    sleep = time.sleep
    monkeypatch.setattr(serve.time, "sleep", lambda _: sleep(0.1))
    monkeypatch.chdir(tmp_path)
    return script


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_returns_once_ready_and_leaves_the_server_running(fake_vllm, capsys) -> None:
    port = _free_port()
    serve.main(["some/model", "--port", str(port), "--ready-timeout", "30"])

    out = capsys.readouterr().out
    assert out.startswith(f"ready: http://127.0.0.1:{port}/v1 (log: vllm-{port}.log;")
    pgid = int(out.rsplit("kill -- -", 1)[1].rstrip(")\n"))
    try:
        assert httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"][0]["id"] == "served"
    finally:
        os.killpg(pgid, signal.SIGTERM)


def test_serve_fails_when_vllm_exits(fake_vllm) -> None:
    fake_vllm.write_text("#!/bin/sh\necho 'CUDA out of memory'\nexit 2\n")
    port = _free_port()
    with pytest.raises(SystemExit, match=f"exited with code 2 before becoming ready; see vllm-{port}"):
        serve.main(["some/model", "--port", str(port), "--ready-timeout", "30"])
    assert "CUDA out of memory" in (fake_vllm.parents[1] / f"vllm-{port}.log").read_text()

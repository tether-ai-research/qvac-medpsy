"""Golden end-to-end pins for every suite factory. Expected values are literals in
``golden_values.py``: pinned reference values; do not regenerate them to make a failing test
pass.

Set ``GOLDEN_DUMP_DIR`` to write the full (un-hashed) payload of each case for diffing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from golden_harness import FakeRemote, canonical, responder
from golden_values import EXPECTED
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

CLOSED_V1 = ("medxpertqa", "AfriMedQA", "mmlu_pro_health", "MedQA_US", "medmcqa", "PubMedQA",
             "mmlu")
UNCAPPED = ("closed_ended_v1/mmlu",)


def _cases() -> list[tuple[str, str, dict]]:
    cases = [(f"closed_ended_v1/{d}", "closed_ended_v1", {"dataset": d}) for d in CLOSED_V1]
    cases.append(("healthbench_v1", "healthbench_v1", {}))
    return cases


def _build(name: str, factory: str, args: dict, remote: FakeRemote):
    from medpsy_eval import _registry

    remote.calls.clear()
    task = getattr(_registry, factory)(**args)
    calls = list(remote.calls)
    remote.calls.clear()
    return task, calls


def _run(tasks, model, roles, log_dir: Path, limit: int | None):
    return inspect_eval(tasks, model=model, model_roles=roles, log_dir=str(log_dir),
                        limit=limit, display="none", log_realtime=False, max_tasks=4)


@pytest.fixture(scope="module")
def golden_payloads(tmp_path_factory) -> dict[str, dict]:
    workdir = tmp_path_factory.mktemp("golden")
    monkeypatch = pytest.MonkeyPatch()
    remote = FakeRemote(workdir)
    remote.install(monkeypatch)
    try:
        model = get_model("mockllm/model", custom_outputs=responder("model"), memoize=False)
        grader = get_model("mockllm/model", custom_outputs=responder("grader", judge=True),
                           memoize=False)
        built = [(name, *_build(name, factory, args, remote))
                 for name, factory, args in _cases()]
        payloads = {}
        # pooled MMLU runs uncapped so every subject is pinned
        for limit, group in ((4, [b for b in built if b[0] not in UNCAPPED]),
                             (None, [b for b in built if b[0] in UNCAPPED])):
            logs = _run([task for _, task, _ in group], model, {"grader": grader},
                        workdir / "logs", limit=limit)
            payloads.update({name: canonical(log, calls)
                             for (name, _, calls), log in zip(group, logs, strict=True)})
        assert remote.calls == [], remote.calls
    finally:
        monkeypatch.undo()

    dump = os.environ.get("GOLDEN_DUMP_DIR")
    if dump:
        Path(dump).mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            Path(dump, name.replace("/", "__") + ".json").write_text(
                json.dumps(payload, indent=1, ensure_ascii=False, default=str))
    return payloads


def _public(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def test_golden_case_set_is_complete(golden_payloads) -> None:
    assert sorted(golden_payloads) == sorted(EXPECTED)


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_golden(case: str, golden_payloads) -> None:
    assert _public(golden_payloads[case]) == EXPECTED[case]

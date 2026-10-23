"""Preset-driven ``medpsy-eval`` CLI: task construction, model-role routing and errors. Tasks are
built offline; ``inspect_ai.eval`` is captured."""

from __future__ import annotations

from argparse import Namespace

import inspect_ai
import inspect_ai.model
import pytest
from inspect_ai._util.registry import registry_info

from medpsy_eval import cli, presets

JUDGE_URL = "http://judge.invalid/v1"
GENERATION = {"temperature": 0.6, "top_p": 0.95,
              "extra_body": {"top_k": 20, "min_p": 0.0,
                             "chat_template_kwargs": {"enable_thinking": True}}}


@pytest.fixture
def captured(monkeypatch) -> dict:
    """``inspect_ai.eval`` and judge ``get_model`` calls, recorded instead of executed."""
    calls: dict = {}

    def fake_eval(tasks, **kwargs):
        calls.update(tasks=tasks, kwargs=kwargs)
        return []

    def fake_get_model(name, base_url=None, config=None):
        return ("model", name, base_url)

    monkeypatch.setattr(inspect_ai, "eval", fake_eval)
    monkeypatch.setattr(inspect_ai.model, "get_model", fake_get_model)
    return calls


def _run(*argv: str) -> None:
    cli.main(["eval", "--model", "mockllm/model", *argv])


def _plan(preset=None, suite=None, datasets="", T=None) -> cli.EvalPlan:
    return cli.plan_eval(Namespace(preset=preset, suite=suite, datasets=datasets, T=T))


@pytest.mark.parametrize(("suite", "names", "judge", "epochs"), [
    ("closed_ended",
     ["medxpertqa", "AfriMedQA", "mmlu_pro_health", "MedQA_US", "medmcqa", "PubMedQA", "mmlu"],
     "vllm/Qwen/QwQ-32B", 3),
    ("healthbench", ["healthbench"], "vllm/opencompass/CompassJudger-2-32B-Instruct", 1),
])
def test_preset_suite_runs_with_grader_judge(suite, names, judge, epochs, captured,
                                             fake_remote) -> None:
    _run("--preset", "medpsy", "--suite", suite, "--judge-base-url", JUDGE_URL)

    assert [t.name for t in captured["tasks"]] == names
    assert {t.epochs for t in captured["tasks"]} == {epochs}
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "mockllm/model"
    assert kwargs["model_base_url"] is None
    assert kwargs["model_roles"] == {"grader": ("model", judge, JUDGE_URL)}
    assert kwargs["max_tokens"] == 16384
    assert {k: kwargs[k] for k in GENERATION} == GENERATION
    assert "top_k" not in kwargs
    assert kwargs["metadata"] == {"judge": presets.compose_suite("medpsy", suite).judge}
    assert set(kwargs["metadata"]["judge"]) == {"model", "revision"}


def test_user_args_override_the_preset() -> None:
    plan = _plan("medpsy", "closed_ended", datasets="mmlu", T=["judge_max_tokens=99"])
    assert plan.datasets == ["mmlu"]
    assert plan.preset_args == {"epochs": 3, "judge_temperature": 0.01, "judge_max_tokens": 4096}
    assert plan.user_args == {"judge_max_tokens": 99}


def test_single_variant_suite_needs_no_dataset(captured, fake_remote) -> None:
    _run("--preset", "medpsy", "--suite", "healthbench", "--judge", "mockllm/model",
         "-T", "epochs=2")
    assert [t.name for t in captured["tasks"]] == ["healthbench"]
    assert captured["tasks"][0].metadata == {"suite": "healthbench_v1"}
    assert captured["tasks"][0].epochs == 2
    assert captured["kwargs"]["model_roles"] == {"grader": ("model", "mockllm/model", None)}
    assert captured["kwargs"]["metadata"] == {"judge": {"model": "mockllm/model"}}


def test_suite_without_preset(captured, fake_remote) -> None:
    _run("--suite", "closed_ended_v1", "--datasets", "MedQA_US,mmlu", "-T", "epochs=1")
    assert [t.name for t in captured["tasks"]] == ["MedQA_US", "mmlu"]
    assert [t.metadata for t in captured["tasks"]] == [
        {"suite": "closed_ended_v1", "protocol": "v1"}] * 2
    assert captured["kwargs"]["model_roles"] is None
    assert captured["kwargs"]["temperature"] is None
    assert captured["kwargs"]["metadata"] is None


def test_unit_runs_standalone_with_grader_judge(captured, fake_remote) -> None:
    _run("--suite", "mmlu", "--judge", "mockllm/judge", "--temperature", "0.6",
         "-T", "epochs=1", "-T", "judge={temperature: 0.2, max_tokens: 77}")
    (task,) = captured["tasks"]
    assert task.name == "mmlu"
    assert task.metadata == {"protocol": "v1"}
    assert task.epochs == 1
    assert [registry_info(s).name for s in task.scorer] == ["medpsy_eval/judge_extraction",
                                                            "medpsy_eval/generation_stats"]
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "mockllm/model"
    assert kwargs["model_roles"] == {"grader": ("model", "mockllm/judge", None)}
    assert kwargs["temperature"] == 0.6
    assert kwargs["metadata"] == {"judge": {"model": "mockllm/judge"}}

    _run("--suite", "medqa")
    assert [t.name for t in captured["tasks"]] == ["MedQA_US"]
    assert captured["kwargs"]["model_roles"] is None


def test_judge_resolution() -> None:
    """Preset judges get ``vllm/`` only with ``--judge-base-url`` (``openai/`` names are kept);
    without it the name is passed to Inspect unchanged; ``--judge`` overrides the preset."""
    def roles(*argv):
        calls = {}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cli.EvalPlan, "build_tasks", lambda self: [])
            mp.setattr(inspect_ai, "eval", lambda tasks, **kw: calls.update(kw) or [])
            mp.setattr(inspect_ai.model, "get_model", lambda name, base_url=None, config=None:
                       (name, base_url))
            _run("--preset", "medpsy", "--suite", "closed_ended", *argv)
        return calls["model"], calls["model_roles"]

    assert roles() == ("mockllm/model", {"grader": ("Qwen/QwQ-32B", None)})
    assert roles("--judge-base-url", JUDGE_URL) == (
        "mockllm/model", {"grader": ("vllm/Qwen/QwQ-32B", JUDGE_URL)})
    assert roles("--judge", "openai/x", "--judge-base-url", JUDGE_URL) == (
        "mockllm/model", {"grader": ("openai/x", JUDGE_URL)})
    assert roles("--no-score") == ("mockllm/model", None)


@pytest.mark.parametrize(("argv", "message"), [
    (("--preset", "medpsy", "--suite", "nope"), "preset 'medpsy' has no suite 'nope'"),
    (("--suite", "nope", "--datasets", "MedQA_US"), "unknown task 'nope'"),
    (("--preset", "medpsy", "--suite", "closed_ended", "--datasets", "MedQA_US,nope"),
     "closed_ended_v1: unknown dataset 'nope'"),
    (("--preset", "medpsy", "--suite", "healthbench", "--datasets", "healthbench_hard"),
     "healthbench_v1: unknown dataset 'healthbench_hard'"),
    (("--suite", "closed_ended_v1"), "closed_ended_v1 needs --datasets or a preset dataset list"),
    (("--preset", "medpsy", "--suite", "closed_ended", "-T", "seed=1"),
     r"closed_ended_v1 has no task argument 'seed'; allowed: \['epochs', 'judge_temperature'"),
    (("--preset", "medpsy", "--suite", "closed_ended", "-T", "dataset=MedQA_US"),
     "select variants with --datasets"),
    (("--preset", "medpsy", "--suite", "closed_ended", "-T", "epochs"),
     "-T expects key=value, got 'epochs'"),
    (("--suite", "mmlu", "-T", "subject=anatomy"),
     r"mmlu has no task argument 'subject'; allowed: \['protocol', 'epochs', 'judge'"),
    (("--suite", "medqa", "-T", "protocol=v2"), r"unknown protocol 'v2'; expected one of"),
    (("--suite", "medqa", "--datasets", "MedQA_US"), "medqa has no variants; drop --datasets"),
    (("--suite", "nope"), r"unknown task 'nope'; expected a suite \('closed_ended_v1'"),
])
def test_invalid_runs_fail_before_building(argv, message, captured, fake_remote) -> None:
    with pytest.raises(SystemExit, match=message):
        _run(*argv)
    assert captured == {}


def test_top_k_is_sent_in_the_request_body(captured, fake_remote) -> None:
    _run("--preset", "medpsy", "--suite", "closed_ended", "--datasets", "MedQA_US",
         "--top-k", "40")
    assert captured["kwargs"]["extra_body"]["top_k"] == 40
    assert "top_k" not in captured["kwargs"]


def test_unknown_preset_key_fails_the_cli(monkeypatch, captured) -> None:
    load = presets.load_preset

    def with_seed(name: str) -> dict:
        preset = load(name)
        preset["suites"]["closed_ended"]["seed"] = 1
        return preset

    monkeypatch.setattr(presets, "load_preset", with_seed)
    with pytest.raises(SystemExit, match=r"medpsy/closed_ended: unknown key\(s\) \['seed'\]"):
        _run("--preset", "medpsy", "--suite", "closed_ended")


def test_unknown_preset_fails_the_cli() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "--preset", "medpsy9", "--suite", "closed_ended"])


def test_preset_requires_suite() -> None:
    with pytest.raises(SystemExit):
        cli.main(["eval", "--preset", "medpsy"])

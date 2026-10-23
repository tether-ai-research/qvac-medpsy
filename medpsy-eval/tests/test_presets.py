"""Preset YAML -> suite composition: flat factory arguments, metadata, and key validation."""

from __future__ import annotations

import copy

import pytest

from medpsy_eval.presets import PRESET_NAMES, compose_suite, load_preset, resolve_suite
from medpsy_eval.suites import SUITES

TASK_ARGS = {
    ("medpsy", "closed_ended"): (
        "closed_ended_v1", {"epochs": 3, "judge_temperature": 0.01, "judge_max_tokens": 4096}),
    ("medpsy", "healthbench"): (
        "healthbench_v1", {"epochs": 1, "judge_temperature": 0.5, "judge_max_tokens": 2048}),
}
JUDGES = {
    ("medpsy", "closed_ended"): ("Qwen/QwQ-32B", "976055f8c83f394f35dbd3ab09a285a984907bd0"),
    ("medpsy", "healthbench"): ("opencompass/CompassJudger-2-32B-Instruct",
                                 "1e586e14ba945e073fb60d73e91ce798e323341c"),
}
DATASETS = {
    ("medpsy", "closed_ended"): (
        "medxpertqa", "AfriMedQA", "mmlu_pro_health", "MedQA_US", "medmcqa", "PubMedQA",
        "mmlu"),
    ("medpsy", "healthbench"): (),
}


def test_preset_names_load() -> None:
    assert tuple(load_preset(name)["name"] for name in PRESET_NAMES) == PRESET_NAMES
    assert {(p, s) for p in PRESET_NAMES for s in load_preset(p)["suites"]} == set(TASK_ARGS)


@pytest.mark.parametrize(("preset", "suite"), TASK_ARGS)
def test_suite_composition(preset: str, suite: str) -> None:
    run = compose_suite(preset, suite)
    task, args = TASK_ARGS[preset, suite]
    assert run.task == task
    assert run.task_args == args
    assert run.datasets == DATASETS[preset, suite]
    assert set(run.task_args) <= set(SUITES[task].arguments)
    assert set(run.datasets) <= set(SUITES[task].datasets)
    model, revision = JUDGES[preset, suite]
    assert run.judge == {"model": model, "revision": revision}


def test_generation_defaults() -> None:
    assert compose_suite("medpsy", "closed_ended").generation == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384,
        "extra_body": {"min_p": 0.0, "chat_template_kwargs": {"enable_thinking": True}},
    }


def _edited(edit) -> dict:
    preset = copy.deepcopy(load_preset("medpsy"))
    edit(preset)
    return preset


@pytest.mark.parametrize(("suite", "edit", "message"), [
    ("closed_ended", lambda p: p["suites"]["closed_ended"].update(seed=1),
     (r"medpsy/closed_ended: unknown key\(s\) \['seed'\]; allowed: \['task', 'datasets', "
      r"'judge', 'epochs', 'judge_temperature'")),
    ("closed_ended", lambda p: p["suites"]["closed_ended"].update(report={}),
     r"medpsy/closed_ended: unknown key\(s\) \['report'\]"),
    ("healthbench", lambda p: p["suites"]["healthbench"]["judge"].update(temperature=0.5),
     (r"medpsy/healthbench judge: unknown key\(s\) \['temperature'\]; "
      r"allowed: \['model', 'revision'\]")),
    ("closed_ended", lambda p: p["generation"].update(seed=1),
     r"medpsy generation: unknown key\(s\) \['seed'\]"),
    ("closed_ended", lambda p: p.update(judge={}),
     r"preset medpsy: unknown key\(s\) \['judge'\]"),
])
def test_unknown_keys_are_errors(suite: str, edit, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_suite(_edited(edit), suite)


@pytest.mark.parametrize(("suite", "edit", "message"), [
    ("healthbench", lambda p: p["suites"]["healthbench"].update(task="healthbench_v3"),
     r"medpsy/healthbench: unknown task 'healthbench_v3'"),
    ("closed_ended", lambda p: p["suites"]["closed_ended"]["datasets"].append("nope"),
     r"medpsy/closed_ended: closed_ended_v1: unknown dataset 'nope'"),
    ("healthbench", lambda p: p["suites"]["healthbench"].update(datasets=["healthbench_hard"]),
     r"medpsy/healthbench: healthbench_v1: unknown dataset 'healthbench_hard'"),
    ("healthbench", lambda p: p["suites"]["healthbench"].pop("task"),
     r"medpsy/healthbench: missing key 'task'"),
    ("nope", lambda p: None, r"preset 'medpsy' has no suite 'nope'"),
])
def test_invalid_suites_are_errors(suite: str, edit, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_suite(_edited(edit), suite)


def test_unknown_preset() -> None:
    with pytest.raises(ValueError, match="unknown preset 'medpsy9'"):
        compose_suite("medpsy9", "closed_ended")



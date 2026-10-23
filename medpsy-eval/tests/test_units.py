"""Benchmark units as standalone Inspect tasks: defaults, Task names and scorers when built on
their own (remote reads faked), Inspect construction by registry name, and selector errors."""

from __future__ import annotations

import inspect

import pytest
from inspect_ai._eval.registry import task_create
from inspect_ai._util.registry import registry_info, registry_params
from inspect_ai.scorer._scorer import scorer_metrics

from medpsy_eval.judging import JudgeSettings, judge_settings
from medpsy_eval.suites import SUITES
from medpsy_eval.units import UNITS

CLOSED = {"protocol": "v1", "epochs": 3, "judge": JudgeSettings(temperature=0.01, max_tokens=4096),
          "tokenizer": None}
DEFAULTS = {
    "afrimedqa": CLOSED,
    "healthbench": {"epochs": 1, "judge": JudgeSettings(temperature=0.5, max_tokens=8192),
                    "tokenizer": None},
    "medmcqa": CLOSED,
    "medqa": CLOSED,
    "medxpertqa": CLOSED,
    "mmlu": CLOSED,
    "mmlu_pro_health": CLOSED,
    "pubmedqa": CLOSED,
}
CLOSED_SCORERS = ["judge_extraction", "generation_stats"]
# unit -> (Task name, epochs, scorers)
STANDALONE = {
    "afrimedqa": ("AfriMedQA", 3, CLOSED_SCORERS),
    "healthbench": ("healthbench", 1, ["healthbench_rubric", "generation_stats"]),
    "medmcqa": ("medmcqa", 3, CLOSED_SCORERS),
    "medqa": ("MedQA_US", 3, CLOSED_SCORERS),
    "medxpertqa": ("medxpertqa", 3, CLOSED_SCORERS),
    "mmlu": ("mmlu", 3, CLOSED_SCORERS),
    "mmlu_pro_health": ("mmlu_pro_health", 3, CLOSED_SCORERS),
    "pubmedqa": ("PubMedQA", 3, CLOSED_SCORERS),
}


def _scorer_names(task) -> list[str]:
    return [registry_info(s).name.split("/")[-1] for s in task.scorer or []]


def _metric_names(metrics) -> list[str]:
    return [registry_info(m).name.split("/")[-1] for m in metrics]


def test_unit_table() -> None:
    assert sorted(DEFAULTS) == sorted(STANDALONE) == sorted(UNITS)
    assert {name: u.judge for name, u in UNITS.items()} == dict.fromkeys(UNITS, "grader")
    assert [name for name, u in UNITS.items() if u.per_dataset] == []


@pytest.mark.parametrize("unit", DEFAULTS)
def test_unit_defaults(unit: str) -> None:
    params = UNITS[unit].parameters.values()
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)
    assert {p.name: p.default for p in params} == DEFAULTS[unit]


@pytest.mark.parametrize("unit", STANDALONE)
def test_unit_builds_standalone(unit: str, fake_remote) -> None:
    name, epochs, scorers = STANDALONE[unit]
    task = UNITS[unit].factory()
    assert task.name == name
    assert task.metadata == ({"protocol": "v1"} if scorers == CLOSED_SCORERS else None)
    assert registry_info(task).name == f"medpsy_eval/{unit}"
    assert task.epochs == epochs
    assert len(task.dataset) > 0
    assert _scorer_names(task) == scorers
    assert task.metrics is None


@pytest.mark.parametrize("unit", [u for u, (_, _, s) in STANDALONE.items() if s == CLOSED_SCORERS])
def test_judge_extraction_reports_answer_rates(unit: str, fake_remote) -> None:
    judge = UNITS[unit].factory().scorer[0]
    names = _metric_names(scorer_metrics(judge))
    assert names[:2] == ["accuracy", "stderr"]
    assert names[-2:] == ["no_answer_rate", "multiple_answers_rate"]


def test_inspect_builds_a_unit_by_registry_name(fake_remote) -> None:
    task = task_create("medpsy_eval/mmlu", epochs=1)
    assert task.name == "mmlu"
    assert task.epochs == 1
    assert _scorer_names(task) == CLOSED_SCORERS


def test_mmlu_pools_the_subjects(fake_remote) -> None:
    from medpsy_eval.benchmarks.mmlu import SUBJECTS, mmlu

    task = mmlu(epochs=1)
    assert [s.id for s in task.dataset] == [f"{subject}/{i}" for subject in SUBJECTS
                                            for i in range(2)]
    first = task.dataset[0].metadata
    assert first["subject"] == "anatomy"
    assert first["dataset"] == "lukaemon_mmlu_anatomy"
    judge, stats = task.scorer
    assert _metric_names(scorer_metrics(judge)) == [
        "accuracy", "stderr", "grouped", "no_answer_rate", "multiple_answers_rate"]
    assert isinstance(scorer_metrics(stats), dict)


def test_tokenizer_reaches_generation_stats(fake_remote) -> None:
    task = SUITES["closed_ended_v1"].factory(dataset="MedQA_US", tokenizer="qvac/MedPsy-4B")
    assert registry_params(task.scorer[1]) == {"tokenizer": "qvac/MedPsy-4B"}


def test_suite_keeps_its_registry_identity_over_the_unit(fake_remote) -> None:
    task = SUITES["closed_ended_v1"].factory(dataset="MedQA_US")
    assert registry_info(task).name == "medpsy_eval/closed_ended_v1"


def test_judge_settings_accept_a_mapping(fake_remote) -> None:
    settings = JudgeSettings(temperature=0.2, max_tokens=77)
    assert judge_settings({"temperature": 0.2, "max_tokens": 77}) == settings
    assert judge_settings(settings) is settings
    task = UNITS["medqa"].factory(judge={"temperature": 0.2, "max_tokens": 77})
    assert _scorer_names(task) == CLOSED_SCORERS


def test_unknown_protocol_is_rejected(fake_remote) -> None:
    message = r"unknown protocol 'v2'; expected one of \('v1',\)"
    with pytest.raises(ValueError, match=message):
        UNITS["medqa"].factory(protocol="v2")
    with pytest.raises(ValueError, match=message):
        task_create("medpsy_eval/medqa", protocol="v2")
    assert fake_remote.calls == []

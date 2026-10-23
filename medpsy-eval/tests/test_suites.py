"""Suite -> benchmark variant composition: variant tables, Task names and metadata, default
arguments, loader/scorer wiring and errors. Tasks are built offline (remote reads are faked)."""

from __future__ import annotations

import importlib
import inspect

import pytest
from inspect_ai._util.registry import registry_info

from medpsy_eval.suites import SUITES

CLOSED_UNITS = {"medxpertqa": "medxpertqa", "AfriMedQA": "afrimedqa",
                "mmlu_pro_health": "mmlu_pro_health", "MedQA_US": "medqa", "medmcqa": "medmcqa",
                "PubMedQA": "pubmedqa", "mmlu": "mmlu"}

# suite -> (suite module, variant table) and variant id -> (unit, constructor, kwargs)
TABLES = {
    "closed_ended_v1": ("closed_ended_v1", "VARIANTS", {
        d: (unit, unit, {"protocol": "v1"}) for d, unit in CLOSED_UNITS.items()
    }),
    "healthbench_v1": ("healthbench_v1", "VARIANTS", {
        "healthbench": ("healthbench", "healthbench", {}),
    }),
}
NAMES = {"closed_ended_v1": "{}", "healthbench_v1": "healthbench"}
PROTOCOLS = {"closed_ended_v1": "v1"}
DEFAULTS = {
    "closed_ended_v1": {"epochs": 3, "judge_temperature": 0.01, "judge_max_tokens": 4096,
                        "tokenizer": None},
    "healthbench_v1": {"epochs": 1, "judge_temperature": 0.5, "judge_max_tokens": 2048,
                       "tokenizer": None},
}
EPOCHS = {"closed_ended_v1": 3, "healthbench_v1": 1}
REMOTE = {
    "medxpertqa": "TsinghuaC3I/MedXpertQA", "AfriMedQA": "afrimedqa/afrimedqa_v2",
    "mmlu_pro_health": "TIGER-Lab/MMLU-Pro", "MedQA_US": "xuxuxuxuxu/MedQA_US_test",
    "medmcqa": "openlifescienceai/medmcqa", "PubMedQA": "bigbio/pubmed_qa",
    "mmlu": "cais/mmlu", "healthbench": "huihuixu/healthbench",
}
SCORERS = {"closed_ended_v1": "judge_extraction", "healthbench": "healthbench_rubric"}


def _variant_cases() -> list[tuple[str, str]]:
    return [(suite, dataset) for suite, (_, _, table) in TABLES.items() for dataset in table]


def _scorer_names(task) -> list[str]:
    return [registry_info(s).name.split("/")[-1] for s in task.scorer or []]


@pytest.mark.parametrize("suite_name", TABLES)
def test_variant_table(suite_name: str) -> None:
    module_name, attr, expected = TABLES[suite_name]
    table = getattr(importlib.import_module(f"medpsy_eval.suites.{module_name}"), attr)
    assert tuple(table) == tuple(expected) == SUITES[suite_name].datasets
    for dataset, (unit, constructor, kwargs) in expected.items():
        variant = table[dataset]
        unit_module = importlib.import_module(f"medpsy_eval.benchmarks.{unit}")
        assert variant.constructor is getattr(unit_module, constructor), dataset
        assert dict(variant.kwargs) == kwargs


@pytest.mark.parametrize("suite_name", DEFAULTS)
def test_factory_defaults(suite_name: str) -> None:
    params = SUITES[suite_name].parameters.values()
    assert {p.name: p.default for p in params if p.default is not inspect.Parameter.empty} == (
        DEFAULTS[suite_name])


def test_judge_roles_and_single_variant_suites() -> None:
    assert {name: s.judge for name, s in SUITES.items()} == {
        "closed_ended_v1": "grader", "healthbench_v1": "grader"}
    assert [name for name, s in SUITES.items() if not s.per_dataset] == ["healthbench_v1"]


@pytest.mark.parametrize(("suite_name", "dataset"), _variant_cases())
def test_variant_builds_offline(suite_name: str, dataset: str, fake_remote) -> None:
    suite = SUITES[suite_name]
    task = suite.factory(dataset=dataset) if suite.per_dataset else suite.factory()
    assert task.name == NAMES[suite_name].format(dataset)
    protocol = {"protocol": PROTOCOLS[suite_name]} if suite_name in PROTOCOLS else {}
    assert task.metadata == {"suite": suite_name, **protocol}
    assert task.epochs == EPOCHS[suite_name]
    assert len(task.dataset) > 0
    assert any(REMOTE[dataset] in call for call in fake_remote.calls), fake_remote.calls
    expected_scorers = [SCORERS[k] for k in (suite_name, dataset) if k in SCORERS]
    assert _scorer_names(task) == [*expected_scorers, "generation_stats"]


def test_suite_arguments_reach_the_constructor(fake_remote) -> None:
    task = SUITES["closed_ended_v1"].factory(dataset="MedQA_US", epochs=1)
    assert task.epochs == 1


@pytest.mark.parametrize("suite_name", [n for n, s in SUITES.items() if s.per_dataset])
def test_unknown_variant_is_rejected(suite_name: str) -> None:
    with pytest.raises(ValueError, match="unknown dataset 'nope'; expected one of"):
        SUITES[suite_name].factory(dataset="nope")


def test_unknown_suite_arguments_are_rejected(fake_remote) -> None:
    with pytest.raises(TypeError):
        SUITES["closed_ended_v1"].factory(dataset="MedQA_US", seed=1)

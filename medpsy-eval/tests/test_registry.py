"""Inspect discovery of the suite factories and benchmark units, the suite registry, and canonical
imports of the benchmark units."""

import importlib
import inspect
from importlib.metadata import entry_points

import pytest
from inspect_ai._util.entrypoints import ensure_entry_points
from inspect_ai._util.registry import is_registry_object, registry_info, registry_lookup

from medpsy_eval import _registry
from medpsy_eval.suites import SUITES, get_suite
from medpsy_eval.units import UNITS, get_task

FACTORIES = (
    "closed_ended_v1",
    "healthbench_v1",
)
UNITS_BY_PACKAGE = {
    "afrimedqa": ("afrimedqa",),
    "healthbench": ("healthbench",),
    "medmcqa": ("medmcqa",),
    "medqa": ("medqa",),
    "medxpertqa": ("medxpertqa",),
    "mmlu": ("mmlu",),
    "mmlu_pro_health": ("mmlu_pro_health",),
    "pubmedqa": ("pubmedqa",),
}
UNIT_TASKS = tuple(sorted(name for names in UNITS_BY_PACKAGE.values() for name in names))
REMOVED = (
    "medpsy_eval.closed_ended",
    "medpsy_eval.datasets",
    "medpsy_eval.prompts",
    "medpsy_eval.scorers",
    "medpsy_eval.tasks",
)


def test_entry_point_targets_the_registry() -> None:
    values = [ep.value for ep in entry_points(group="inspect_ai") if ep.name == "medpsy_eval"]
    assert values == ["medpsy_eval._registry"]


def test_registry_exposes_exactly_the_suite_factories_and_units() -> None:
    assert sorted(_registry.__all__) == sorted((*FACTORIES, *UNIT_TASKS))
    assert sorted(SUITES) == list(FACTORIES)
    assert sorted(UNITS) == list(UNIT_TASKS)
    for name in FACTORIES:
        assert getattr(_registry, name) is SUITES[name].factory is get_suite(name).factory
        assert get_task(name) is SUITES[name]
        assert SUITES[name].name == name
    for name in UNIT_TASKS:
        assert getattr(_registry, name) is UNITS[name].factory is get_task(name).factory
        assert UNITS[name].name == name


@pytest.mark.parametrize("name", (*FACTORIES, *UNIT_TASKS))
def test_inspect_discovers_factory(name: str) -> None:
    ensure_entry_points("medpsy_eval")
    factory = registry_lookup("task", f"medpsy_eval/{name}")
    assert factory is getattr(_registry, name)
    assert registry_info(factory).type == "task"
    assert registry_info(factory).name == f"medpsy_eval/{name}"


def test_unknown_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task 'nope'; expected one of"):
        get_suite("nope")
    with pytest.raises(ValueError, match="unknown task 'medqa'"):
        get_suite("medqa")
    with pytest.raises(ValueError, match=r"unknown task 'nope'; expected a suite \('closed_ended_v1'"):
        get_task("nope")


@pytest.mark.parametrize(("unit", "constructors"), UNITS_BY_PACKAGE.items())
def test_benchmark_unit_owns_its_implementation(unit: str, constructors: tuple[str]) -> None:
    module = importlib.import_module(f"medpsy_eval.benchmarks.{unit}")
    importlib.import_module(f"medpsy_eval.benchmarks.{unit}.dataset")
    importlib.import_module(f"medpsy_eval.benchmarks.{unit}.prompts")
    for name in constructors:
        constructor = getattr(module, name)
        assert name in module.__all__
        assert is_registry_object(constructor)
        params = inspect.signature(constructor).parameters.values()
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params), name


@pytest.mark.parametrize("module", REMOVED)
def test_horizontal_modules_and_facades_are_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


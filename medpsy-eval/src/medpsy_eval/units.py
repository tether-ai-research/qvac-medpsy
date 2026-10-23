"""Benchmark units run on their own (``medpsy-eval eval --suite medqa``, ``inspect eval
medpsy_eval/medqa``). Units take no ``--datasets``."""

from __future__ import annotations

from .benchmarks.afrimedqa import afrimedqa
from .benchmarks.healthbench import healthbench
from .benchmarks.medmcqa import medmcqa
from .benchmarks.medqa import medqa
from .benchmarks.medxpertqa import medxpertqa
from .benchmarks.mmlu import mmlu
from .benchmarks.mmlu_pro_health import mmlu_pro_health
from .benchmarks.pubmedqa import pubmedqa
from .suites import SUITES, Suite

UNITS: dict[str, Suite] = {
    unit.name: unit
    for unit in (
        Suite(afrimedqa, ()),
        Suite(healthbench, ()),
        Suite(medmcqa, ()),
        Suite(medqa, ()),
        Suite(medxpertqa, ()),
        Suite(mmlu, ()),
        Suite(mmlu_pro_health, ()),
        Suite(pubmedqa, ()),
    )
}


def get_task(name: str) -> Suite:
    """A suite factory or a benchmark unit by registry name."""
    try:
        return SUITES.get(name) or UNITS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; expected a suite {tuple(SUITES)} "
                         f"or a unit {tuple(UNITS)}") from None


__all__ = ["UNITS", "get_task"]

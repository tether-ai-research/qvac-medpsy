"""Evaluation presets: which suites make up a paper's evaluation, with the run settings for
each. Server settings are not part of a preset.

A preset suite block holds ``task`` (a suite factory), ``datasets`` (variant ids), ``judge``
(judge identity metadata) and task arguments, each named after a parameter of the suite's task
factory. Any other key is an error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

PRESET_NAMES = ("medpsy",)
PRESET_KEYS = ("name", "generation", "suites")
GENERATION_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "extra_body")
SUITE_KEYS = ("task", "datasets", "judge")
JUDGE_KEYS = ("model", "revision")


@dataclass(frozen=True)
class SuiteComposition:
    """Runnable part of one declarative suite.

    Serving remains an explicit runtime concern; benchmark selection, task arguments, and
    generation defaults come from the preset.
    """

    preset: str
    suite: str
    task: str
    datasets: tuple[str, ...]
    task_args: dict[str, Any]
    generation: dict[str, Any]
    judge: dict[str, Any] | None


def load_preset(name: str) -> dict[str, Any]:
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {name!r}; expected one of {PRESET_NAMES}")
    text = files("medpsy_eval").joinpath("configs", "presets", f"{name}.yaml").read_text()
    return yaml.safe_load(text)


def suite(preset: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return preset["suites"][name]
    except KeyError:
        raise ValueError(f"preset {preset['name']!r} has no suite {name!r}; "
                         f"available: {sorted(preset['suites'])}") from None


def _check_keys(where: str, mapping: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed = tuple(allowed)
    unknown = [key for key in mapping if key not in allowed]
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {list(allowed)}")


def compose_suite(preset_name: str, suite_name: str) -> SuiteComposition:
    """Resolve preset YAML into a task invocation without loading benchmark data."""
    return resolve_suite(load_preset(preset_name), suite_name)


def resolve_suite(preset: Mapping[str, Any], suite_name: str) -> SuiteComposition:
    from .suites import get_suite

    name = preset["name"]
    _check_keys(f"preset {name}", preset, PRESET_KEYS)
    generation = dict(preset.get("generation") or {})
    _check_keys(f"{name} generation", generation, GENERATION_KEYS)
    raw = suite(preset, suite_name)
    where = f"{name}/{suite_name}"
    if "task" not in raw:
        raise ValueError(f"{where}: missing key 'task'")
    try:
        s = get_suite(raw["task"])
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None

    _check_keys(where, raw, (*SUITE_KEYS, *s.arguments))
    judge = dict(raw["judge"]) if raw.get("judge") else None
    if judge:
        _check_keys(f"{where} judge", judge, JUDGE_KEYS)
    datasets = tuple(raw.get("datasets") or ())
    try:
        for dataset in datasets:
            s.check_dataset(dataset)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None

    return SuiteComposition(
        preset=name,
        suite=suite_name,
        task=s.name,
        datasets=datasets,
        task_args={key: raw[key] for key in s.arguments if key in raw},
        generation=generation,
        judge=judge,
    )

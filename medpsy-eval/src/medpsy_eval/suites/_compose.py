"""Suite composition: select a benchmark variant, apply suite arguments, record the suite in the
Task metadata."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from inspect_ai import Task, task_with


class Variant(NamedTuple):
    """``constructor(**kwargs, **suite_args)`` builds the variant's Task."""

    constructor: Callable[..., Task]
    kwargs: Mapping[str, Any] = {}


def _variant(variants: Mapping[str, Variant], dataset: str) -> Variant:
    try:
        return variants[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {tuple(variants)}"
        ) from None


def compose(variants: Mapping[str, Variant], dataset: str, /, *, suite: str,
            **suite_args: Any) -> Task:
    """Build variant ``dataset``; its Task keeps the unit's name and gains metadata ``suite``."""
    variant = _variant(variants, dataset)
    task = variant.constructor(**{**variant.kwargs, **suite_args})
    return task_with(task, metadata={"suite": suite, **(task.metadata or {})})


JudgeRole = Literal["grader"]


@dataclass(frozen=True)
class Suite:
    """A registered task factory (a suite, or a benchmark unit run on its own) and the variant
    ids it composes.

    ``judge`` says where the suite's judge runs: as Inspect's ``grader`` model role, or nowhere
    (``None``).
    """

    factory: Callable[..., Task]
    datasets: tuple[str, ...]
    judge: JudgeRole | None = "grader"

    @classmethod
    def of(cls, factory: Callable[..., Task], variants: Mapping[str, Variant],
           judge: JudgeRole | None = "grader") -> Suite:
        return cls(factory, tuple(variants), judge)

    @property
    def name(self) -> str:
        return self.factory.__name__

    @property
    def parameters(self) -> Mapping[str, inspect.Parameter]:
        return inspect.signature(self.factory, eval_str=True).parameters

    @property
    def per_dataset(self) -> bool:
        """Whether the factory selects a variant with ``dataset`` (single-variant suites
        take no ``dataset`` argument)."""
        return "dataset" in self.parameters

    @property
    def arguments(self) -> tuple[str, ...]:
        """Task arguments a preset or ``-T`` may set (``dataset`` is chosen separately)."""
        return tuple(name for name in self.parameters if name != "dataset")

    def check_dataset(self, dataset: str) -> None:
        if not self.datasets:
            raise ValueError(f"{self.name} has no variants; drop --datasets {dataset!r}")
        if dataset not in self.datasets:
            raise ValueError(
                f"{self.name}: unknown dataset {dataset!r}; expected one of {self.datasets}"
            )

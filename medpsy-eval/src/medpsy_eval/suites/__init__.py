"""Inspect task factories composing MedPsy benchmark implementations."""

from ._compose import Suite
from .closed_ended_v1 import SUITES as _CLOSED_ENDED_V1
from .closed_ended_v1 import closed_ended_v1
from .healthbench_v1 import SUITES as _HEALTHBENCH_V1
from .healthbench_v1 import healthbench_v1

SUITES: dict[str, Suite] = {
    suite.name: suite
    for suite in (
        *_CLOSED_ENDED_V1,
        *_HEALTHBENCH_V1,
    )
}


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; expected one of {tuple(SUITES)}") from None


__all__ = [
    "SUITES",
    "Suite",
    "closed_ended_v1",
    "get_suite",
    "healthbench_v1",
]

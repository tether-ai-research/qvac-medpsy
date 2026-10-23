from .benchmarks.afrimedqa import afrimedqa
from .benchmarks.healthbench import healthbench
from .benchmarks.medmcqa import medmcqa
from .benchmarks.medqa import medqa
from .benchmarks.medxpertqa import medxpertqa
from .benchmarks.mmlu import mmlu
from .benchmarks.mmlu_pro_health import mmlu_pro_health
from .benchmarks.pubmedqa import pubmedqa
from .suites import closed_ended_v1, healthbench_v1

__all__ = [
    "afrimedqa",
    "closed_ended_v1",
    "healthbench",
    "healthbench_v1",
    "medmcqa",
    "medqa",
    "medxpertqa",
    "mmlu",
    "mmlu_pro_health",
    "pubmedqa",
]

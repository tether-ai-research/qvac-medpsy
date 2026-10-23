"""HealthBench rubric benchmark (dataset id ``healthbench``, with the Hard and consensus
subsets as metrics)."""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from ...generation_stats import generation_stats
from ...judging import JudgeArg, JudgeSettings, judge_settings
from ...messages import to_messages
from .dataset import load_healthbench, subset_ids
from .scorer import healthbench_rubric

JUDGE = JudgeSettings(temperature=0.5, max_tokens=8192)
NAME = "healthbench"


@task
def healthbench(
    *,
    epochs: int = 1,
    judge: JudgeArg = JUDGE,
    tokenizer: str | None = None,
) -> Task:
    scorer = healthbench_rubric(
        config=judge_settings(judge).generate_config(),
        hard_ids=frozenset(subset_ids("hard")),
        consensus_ids=frozenset(subset_ids("consensus")),
    )
    samples = [
        Sample(
            id=index,
            input=to_messages(record["prompt"]),
            target=record["prompt_id"],
            metadata={"record": record, "dataset": NAME},
        )
        for index, record in enumerate(load_healthbench(""))
    ]
    return Task(
        name=NAME,
        dataset=MemoryDataset(samples, name=NAME),
        solver=generate(),
        scorer=[scorer, generation_stats(tokenizer)],
        epochs=Epochs(epochs, "mean"),
    )


__all__ = ["healthbench"]

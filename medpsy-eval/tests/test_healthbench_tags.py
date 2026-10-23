"""HealthBench axis/theme tag metrics (the per-tag ``healthbench-*`` results)."""

from __future__ import annotations

import pytest
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import SampleScore, Score, scorer

from medpsy_eval.benchmarks.healthbench.scorer import healthbench_tags, tag_means


def scored(tags: dict[str, float], subsets: list[str]) -> SampleScore:
    return SampleScore(score=Score(value=0.0, metadata={"tag_scores": tags, "subsets": subsets}))


def test_tag_means_per_subset_clipped_and_filtered() -> None:
    scores = [
        scored({"axis:accuracy": 0.5, "theme:hedging": -0.4, "level:example": 1.0,
                "physician_agreed_category:x": 1.0}, ["hard"]),
        scored({"axis:accuracy": 1.0, "cluster:hedging_any": 1.0}, ["consensus"]),
        scored({"axis:completeness": 0.2}, []),
    ]
    assert tag_means(scores) == {
        "healthbench-axis:accuracy": 0.75,
        "healthbench-axis:completeness": 0.2,
        "healthbench-theme:hedging": 0.0,
        "healthbench_hard-axis:accuracy": 0.5,
        "healthbench_hard-theme:hedging": 0.0,
        "healthbench_consensus-axis:accuracy": 1.0,
    }


TAGS_BY_EPOCH = {1: 0.2, 2: 0.6}


@scorer(metrics=[healthbench_tags()])
def epoch_tags():
    async def score(state, target) -> Score:
        value = TAGS_BY_EPOCH[state.epoch]
        return Score(value=value, metadata={"tag_scores": {"axis:accuracy": value},
                                            "subsets": ["hard"]})

    return score


def test_tag_metrics_count_every_epoch(tmp_path) -> None:
    task = Task(dataset=[Sample(input="q", target="t")], scorer=epoch_tags(),
                epochs=Epochs(2, "mean"))
    (log,) = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
    assert metrics == {"healthbench-axis:accuracy": pytest.approx(0.4),
                       "healthbench_hard-axis:accuracy": pytest.approx(0.4)}

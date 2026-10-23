"""Generation diagnostics and the closed-ended answer-rate metrics."""

from __future__ import annotations

import math

import pytest
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, accuracy, scorer

from medpsy_eval.answer_scoring import multiple_answers_rate, no_answer_rate
from medpsy_eval.answer_scoring.extraction import MULTIPLE_ANSWERS, NO_ANSWER
from medpsy_eval.generation_stats import (
    count_reasoning_blocks,
    sample_stats,
    split_reasoning_answer,
)


def words(text: str) -> int:
    return len(text.split())


def test_server_split_reasoning_keeps_the_answer_unstripped() -> None:
    assert split_reasoning_answer("  weigh options \n", "\n\nAnswer: B") == (
        "weigh options", "\n\nAnswer: B")


@pytest.mark.parametrize(("prediction", "expected"), [
    ("<think>weigh</think>\nAnswer: B", ("weigh", "Answer: B")),
    ("weigh</think> B", ("weigh", "B")),
    ("<think>never closed", ("never closed", "")),
    ("[THOUGHT]t[/THOUGHT] A", ("t", "A")),
    ("  plain answer ", ("", "plain answer")),
])
def test_embedded_reasoning_split(prediction: str, expected: tuple[str, str]) -> None:
    assert split_reasoning_answer("", prediction) == expected


@pytest.mark.parametrize(("reasoning", "prediction", "blocks"), [
    ("think", "A", 1),
    ("think", "more</think> A", 2),
    ("think</think>again", "A", 2),
    ("", "<think>t</think> A", 1),
    ("", "<think>never closed", 0),
    ("", "A", 0),
])
def test_reasoning_blocks_count_end_markers(reasoning: str, prediction: str, blocks: int) -> None:
    assert count_reasoning_blocks(reasoning, prediction) == blocks


def test_stats_of_a_split_completion() -> None:
    stats = sample_stats("weigh the options", "\n\nAnswer: B", "stop", words)
    assert stats == {
        "token_len_reasoning": 3.0, "token_len_answer": 2.0, "token_len_total": 5.0,
        "zero_token_answer_rate": 0.0, "zero_reasoning_rate": 0.0,
        "single_reasoning_rate": 100.0, "multiple_reasoning_rate": 0.0,
        "truncation_rate": 0.0, "truncated_in_reasoning_rate": 0.0,
        "truncated_in_answer_rate": 0.0,
    }


def test_unterminated_reasoning_has_no_closed_block_and_an_empty_answer() -> None:
    stats = sample_stats("", "<think>ran out of tokens", "max_tokens", words)
    assert stats["token_len_reasoning"] == 4.0
    assert stats["token_len_answer"] == 0.0
    assert stats["zero_token_answer_rate"] == 100.0
    assert stats["zero_reasoning_rate"] == 0.0
    assert stats["single_reasoning_rate"] == stats["multiple_reasoning_rate"] == 0.0
    assert stats["truncation_rate"] == stats["truncated_in_reasoning_rate"] == 100.0
    assert stats["truncated_in_answer_rate"] == 0.0


def test_truncated_in_the_answer() -> None:
    stats = sample_stats("done thinking", "The answer is", "model_length", words)
    assert stats["truncation_rate"] == stats["truncated_in_answer_rate"] == 100.0
    assert stats["truncated_in_reasoning_rate"] == 0.0


def test_no_reasoning_and_multiple_blocks() -> None:
    assert sample_stats("", "B", "stop", words)["zero_reasoning_rate"] == 100.0
    multi = sample_stats("first", "second</think> B", "stop", words)
    assert multi["multiple_reasoning_rate"] == 100.0
    assert multi["single_reasoning_rate"] == 0.0


def test_unknown_stop_reason_and_missing_tokenizer_leave_keys_unscored() -> None:
    stats = sample_stats("t", "A", "unknown", None)
    unscored = {k for k, v in stats.items() if math.isnan(v)}
    assert unscored == {"token_len_reasoning", "token_len_answer", "token_len_total",
                        "zero_token_answer_rate", "truncation_rate",
                        "truncated_in_reasoning_rate", "truncated_in_answer_rate"}
    assert stats["single_reasoning_rate"] == 100.0


ANSWERS_BY_EPOCH = {1: "A", 2: NO_ANSWER, 3: MULTIPLE_ANSWERS, 4: NO_ANSWER}


@scorer(metrics=[accuracy(), no_answer_rate(), multiple_answers_rate()])
def epoch_answers():
    async def score(state, target) -> Score:
        answer = ANSWERS_BY_EPOCH[state.epoch]
        return Score(value="C" if answer == target.text else "I", answer=answer)

    return score


def test_answer_rates_count_every_epoch(tmp_path) -> None:
    task = Task(dataset=[Sample(input="q", target="A")], scorer=epoch_answers(),
                epochs=Epochs(4, "mean"))
    (log,) = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
    assert metrics == {"accuracy": 0.25, "no_answer_rate": 50.0, "multiple_answers_rate": 25.0}

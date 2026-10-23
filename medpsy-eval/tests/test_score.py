"""``medpsy-eval score``: judging saved generations, one scored log per judge."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model

from medpsy_eval import cli

JUDGE_URL = "http://judge.invalid/v1"


@pytest.fixture
def judges(monkeypatch) -> list[tuple]:
    """``grader_role`` calls, answered by a mock judge that always replies with the letter in
    ``mockllm/judge-<letter>`` (A for any other name) or ``criteria_met: true``."""
    calls: list[tuple] = []

    def fake_grader_role(judge, base_url, max_connections, from_preset):
        calls.append((judge, base_url, max_connections, from_preset))
        letter = judge.rsplit("-", 1)[-1] if judge.startswith("mockllm/judge-") else "A"

        def respond(input, tools, tool_choice, config) -> ModelOutput:
            text = "\n".join(message.text for message in input)
            reply = ('{"explanation": "ok", "criteria_met": true}' if "criteria_met" in text
                     else f"Extracted Candidate's Answer: {letter}")
            return ModelOutput.from_content(model="mockllm", content=reply)

        return {"grader": get_model("mockllm/model", custom_outputs=respond, memoize=False)}

    monkeypatch.setattr(cli, "grader_role", fake_grader_role)
    return calls


def _eval(log_dir: Path, *argv: str) -> Path:
    cli.main(["eval", "--model", "mockllm/model", "--log-dir", str(log_dir), "--display", "none",
              *argv])
    (log,) = log_dir.glob("*.eval")
    return log


def _metrics(log) -> dict:
    return {k: v.value for k, v in log.results.scores[0].metrics.items()}


def test_two_judges_score_one_generation(tmp_path, fake_remote, judges) -> None:
    generations = _eval(tmp_path / "gen", "--suite", "medqa", "--judge", "mockllm/judge-A",
                        "--no-score")

    for letter in "AB":
        cli.main(["score", str(generations), "--judge", f"mockllm/judge-{letter}",
                  "--log-dir", str(tmp_path / "scored"), "--display", "none"])

    assert [j[0] for j in judges] == ["mockllm/judge-A", "mockllm/judge-B"]
    assert not read_eval_log(generations).results.scores
    for letter in "AB":
        out = tmp_path / "scored" / f"{generations.stem}_scored_mockllm_judge-{letter}.eval"
        log = read_eval_log(out)
        assert log.eval.metadata["judge"] == {"model": f"mockllm/judge-{letter}"}
        answers = {s.scores["judge_extraction"].answer for s in log.samples}
        assert answers == {letter}


def test_score_matches_scoring_during_eval(tmp_path, fake_remote, judges) -> None:
    """HealthBench Hard/Consensus come out the same whether scored inline or afterwards: the
    scorers are rebuilt from the task, not from the log's serialized scorer arguments."""
    preset = ("--preset", "medpsy", "--suite", "healthbench")
    inline = read_eval_log(_eval(tmp_path / "inline", *preset))
    generations = _eval(tmp_path / "gen", *preset, "--no-score")

    cli.main(["score", str(generations), "--judge-base-url", JUDGE_URL, "--display", "none"])

    judge = "opencompass/CompassJudger-2-32B-Instruct"
    assert judges == [(judge, None, None, True), (judge, JUDGE_URL, None, True)]
    out = generations.with_name(f"{generations.stem}_scored_opencompass_CompassJudger-2-32B"
                                "-Instruct.eval")
    scored = read_eval_log(out)
    assert scored.eval.metadata["judge"] == inline.eval.metadata["judge"]
    assert _metrics(scored) == _metrics(inline)
    subsets = [k for k in _metrics(scored) if "hard" in k or "consensus" in k]
    assert subsets and not any(math.isnan(_metrics(scored)[k]) for k in subsets)


def test_score_errors(tmp_path, fake_remote, judges) -> None:
    generations = _eval(tmp_path / "gen", "--suite", "medqa", "--no-score")
    with pytest.raises(SystemExit, match="the log records no judge; pass --judge"):
        cli.main(["score", str(generations)])

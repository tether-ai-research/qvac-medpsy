"""``inspect eval-retry`` resumes our logs: the task is rebuilt from its registry name and the
task args recorded in the log."""

from __future__ import annotations

import inspect_ai
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model

from medpsy_eval.suites import SUITES


def test_eval_retry_resumes_an_errored_suite_log(tmp_path, fake_remote) -> None:
    calls = []

    def fail_first(input, tools, tool_choice, config) -> ModelOutput:
        calls.append(input)
        if len(calls) == 1:
            raise RuntimeError("generation failed")
        return ModelOutput.from_content(model="mockllm", content="The answer is A.")

    task = SUITES["closed_ended_v1"].factory(dataset="MedQA_US", epochs=1,
                                            judge_max_tokens=77)
    grader = get_model("mockllm/model", memoize=False)
    model = get_model("mockllm/model", custom_outputs=fail_first, memoize=False)
    (failed,) = inspect_ai.eval(task, model=model, model_roles={"grader": grader},
                                log_dir=str(tmp_path), display="none", max_connections=1)
    assert failed.status == "error"

    (retried,) = inspect_ai.eval_retry(failed.location, log_dir=str(tmp_path), display="none")

    assert retried.status == "success"
    assert retried.eval.task_args == failed.eval.task_args
    assert retried.eval.task_args["judge_max_tokens"] == 77
    assert len(read_eval_log(retried.location).samples) == len(failed.eval.dataset.sample_ids)
    assert {s.scorer for s in retried.results.scores} == {"judge_extraction", "generation_stats"}

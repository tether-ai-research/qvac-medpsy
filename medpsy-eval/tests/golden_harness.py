"""Offline end-to-end harness for the golden tests.

Every suite factory is evaluated through ``inspect_ai.eval`` on synthetic rows, with a
deterministic mock model as both the evaluated model and the judge. Remote reads are faked
at the boundaries that do not depend on package layout: ``datasets.load_dataset`` and
``huggingface_hub.hf_hub_download`` (wherever a module bound them). The canonical payload of each run pins Task names, requested
dataset revisions, rendered inputs, every model call (inference and judge prompts plus request
settings), sample metadata, scores and metrics.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
import huggingface_hub
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ModelOutput,
)

# --- synthetic remote data --------------------------------------------------------------------

def _medxpertqa() -> list[dict]:
    return [
        {"id": f"Text-{i}", "question": f"A {50 + i}-year-old presents with chest pain {i}.",
         "options": {k: f"option {k.lower()}{i}" for k in "ABCDE"}, "label": "CA"[i],
         "medical_task": "Diagnosis", "body_system": "Cardiovascular",
         "question_type": "Reasoning"}
        for i in range(2)
    ]


def _afrimedqa() -> list[dict]:
    def row(i, split, qtype, correct):
        return {"sample_id": f"af{i}", "question": f"Which drug treats malaria {i}?",
                "question_clean": "", "answer_options": json.dumps(
                    {f"option{j + 1}": f"drug {j}-{i}" for j in range(5)}),
                "correct_answer": correct, "split": split, "question_type": qtype,
                "specialty": "Infectious Disease", "country": "NG"}
    return [row(0, "test", "mcq", "option2"), row(1, "train", "mcq", "option1"),
            row(2, "test", "saq", "option1"), row(3, "test", "mcq", "option5")]


def _mmlu_pro() -> list[dict]:
    return [
        {"question_id": i, "question": f"Health question {i}?",
         "options": ["alpha", "beta", "N/A", "delta"] if i == 0 else ["one", "two", "three"],
         "answer": "BD"[i % 2] if i < 2 else "A", "answer_index": 1, "cot_content": "",
         "category": "health" if i != 1 else "law", "src": "ori_mmlu-health"}
        for i in range(3)
    ]


def _medqa() -> list[dict]:
    return [
        {"question": f"A patient with condition {i} needs which treatment?", "answer": "x",
         "options": {k: f"treatment {k}{i}" for k in "ABCDE"}, "meta_info": "step1",
         "answer_idx": "EB"[i]}
        for i in range(2)
    ]


def _medmcqa() -> list[dict]:
    return [
        {"id": f"m{i}", "question": f"Most common site of lesion {i}?", "opa": f"a{i}",
         "opb": f"b{i}", "opc": f"c{i}", "opd": f"d{i}", "cop": i % 4, "choice_type": "single",
         "exp": "", "subject_name": "Social_Preventive", "topic_name": "Epidemiology"}
        for i in range(2)
    ]


def _mmlu(subject: str) -> list[dict]:
    return [
        {"question": f"{subject} question {i}?", "subject": subject,
         "choices": [f"w{i}", f"x{i}", f"y{i}", f"z{i}"], "answer": (i + 1) % 4}
        for i in range(2)
    ]


def _healthbench(subset: str) -> list[dict]:
    rows = [
        {"prompt": [{"role": "user", "content": f"I have a headache {i}, what should I do?"}]
                   + ([{"role": "assistant", "content": "How long?"},
                       {"role": "user", "content": "Two days."}] if i == 1 else []),
         "prompt_id": f"pid-{i}",
         "rubrics": [{"criterion": f"Advises rest {i}", "points": 5,
                      "tags": ["axis:accuracy", "level:example"]},
                     {"criterion": "Recommends dangerous drug", "points": -3,
                      "tags": ["axis:safety", "level:example"]}],
         "example_tags": ["theme:emergency_referrals", f"physician_agreed_category:c{i}"]}
        for i in range(3)
    ]
    return {"": rows, "hard": rows[:1], "consensus": rows[1:2]}[subset]


def hf_rows(path: str, args: tuple, kwargs: dict) -> list[dict]:
    config = args[0] if args else None
    if path == "TsinghuaC3I/MedXpertQA":
        return _medxpertqa()
    if path == "afrimedqa/afrimedqa_v2":
        return _afrimedqa()
    if path == "TIGER-Lab/MMLU-Pro":
        return _mmlu_pro()
    if path == "xuxuxuxuxu/MedQA_US_test":
        return _medqa()
    if path == "openlifescienceai/medmcqa":
        return _medmcqa()
    if path == "cais/mmlu":
        return _mmlu(config)
    if path == "huihuixu/healthbench":
        name = kwargs["data_files"]["test"]
        return _healthbench(next(k for k in ("hard", "consensus") if name.startswith(k))
                            if not name.startswith("2025") else "")
    raise AssertionError(f"unexpected load_dataset({path!r}, {args}, {kwargs})")


def _pubmedqa_zip(directory: Path) -> Path:
    path = directory / "pqal.zip"
    data = {
        str(100 + i): {"QUESTION": f"Does exposure {i} increase risk?",
                       "CONTEXTS": [f"Background {i}.", f"Results {i}."],
                       "final_decision": ["yes", "no", "maybe"][i]}
        for i in range(3)
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("pqal_test_set.json", json.dumps(data))
    return path


# --- fake remote ------------------------------------------------------------------------------

_REAL_LOAD_DATASET = datasets.load_dataset
_REAL_HF_HUB_DOWNLOAD = huggingface_hub.hf_hub_download


def whitespace_tokens(text: str) -> int:
    """Stand-in tokenizer for ``generation_stats`` (the mock model has none)."""
    return len(text.split())


def _describe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _describe(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    return value


@dataclass
class FakeRemote:
    workdir: Path
    calls: list[str] = field(default_factory=list)

    def load_dataset(self, path, *args, **kwargs):
        self.calls.append(json.dumps(["load_dataset", path, _describe(args), _describe(kwargs)]))
        return datasets.Dataset.from_list(hf_rows(path, args, kwargs))

    def hf_hub_download(self, repo_id, filename, **kwargs):
        self.calls.append(json.dumps(["hf_hub_download", repo_id, filename, _describe(kwargs)]))
        assert (repo_id, filename) == ("bigbio/pubmed_qa", "pqal.zip"), (repo_id, filename)
        return str(_pubmedqa_zip(self.workdir))

    def install(self, monkeypatch) -> None:
        import medpsy_eval._registry  # noqa: F401  (imports every module bound to the readers)

        monkeypatch.setattr(datasets, "load_dataset", self.load_dataset)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self.hf_hub_download)
        for name, module in list(sys.modules.items()):
            if not name.startswith("medpsy_eval"):
                continue
            if getattr(module, "load_dataset", None) is _REAL_LOAD_DATASET:
                monkeypatch.setattr(module, "load_dataset", self.load_dataset)
            if getattr(module, "hf_hub_download", None) is _REAL_HF_HUB_DOWNLOAD:
                monkeypatch.setattr(module, "hf_hub_download", self.hf_hub_download)

        from medpsy_eval import generation_stats

        monkeypatch.setattr(generation_stats, "token_counter", lambda name: whitespace_tokens)

        def no_network(*args, **kwargs):
            raise AssertionError(f"network access in golden test: {args} {kwargs}")

        import httpx

        monkeypatch.setattr(httpx, "get", no_network)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")


# --- deterministic mock model -----------------------------------------------------------------

REPLIES: list[str | tuple[str, str]] = [
    "The answer is E.",
    "<think>Weighing the options.</think>\nAnswer: B",
    ("Option C fits best because of the history.", "Final answer: C"),
    "<think>Unterminated reasoning about option A",
    "A",
    "Yes",
    "No, this is not hateful.",
    '{"is_answer_correct": "yes", "answer": "1", "why_correct": "fits"}',
    '{"correct_index": 3, "cop": "None of the above", "why_correct": "none fit"}',
    "1",
    "0",
    "1\n2 Given albuterol.",
    "CORRECT",
    "I cannot help with that request; please consult a clinician.",
    "",
]

GARBLED = "I am not sure how to format this."

# Judge replies by the output format the judge prompt asks for (first marker found wins).
JUDGE_REPLIES: list[tuple[str, list[str | tuple[str, str]]]] = [
    ("Extracted Candidate's Answer", [
        "Extracted Candidate's Answer: A",
        "Extracted Candidate's Answer: E",
        ("Looking for the final answer.", "Extracted Candidate's Answer: B"),
        "Extracted Candidate's Answer: NO_ANSWER",
        "Extracted Candidate's Answer: MULTIPLE_ANSWERS",
        GARBLED,
    ]),
    ("criteria_met", [
        '{"explanation": "Meets the criterion.", "criteria_met": true}',
        '```json\n{"explanation": "Not met.", "criteria_met": false}\n```',
        'Reasoning first. {"explanation": "x", "criteria_met": "true"}',
        GARBLED,
    ]),
]


def _output(reply: str | tuple[str, str]) -> ModelOutput:
    if isinstance(reply, str):
        return ModelOutput.from_content(model="mockllm", content=reply)
    reasoning, text = reply
    message = ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)], model="mockllm")
    return ModelOutput(model="mockllm",
                       choices=[ChatCompletionChoice(message=message, stop_reason="stop")])


def responder(salt: str, judge: bool = False) -> Callable:
    """Reply chosen by a hash of the input; judges pick from the format their prompt asks for."""

    def respond(input, tools, tool_choice, config) -> ModelOutput:
        digest = hashlib.sha256(salt.encode())
        for message in input:
            digest.update(message.role.encode())
            digest.update(message.text.encode())
        pool = REPLIES
        if judge:
            text = "\n".join(message.text for message in input)
            pool = next((replies for marker, replies in JUDGE_REPLIES if marker in text), [GARBLED])
        return _output(pool[int(digest.hexdigest(), 16) % len(pool)])

    return respond


# --- canonical payload ------------------------------------------------------------------------


def _sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=False, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _message(message) -> dict:
    out = {"role": message.role, "text": message.text}
    if not isinstance(message.content, str):
        reasoning = [c.reasoning for c in message.content if c.type == "reasoning"]
        if reasoning:
            out["reasoning"] = reasoning
    return out


def _input(value) -> Any:
    return value if isinstance(value, str) else [_message(m) for m in value]


def _number(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, dict):
        return {k: _number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_number(v) for v in value]
    return value


def canonical(log, remote_calls: list[str]) -> dict[str, Any]:
    """Comparable summary of one eval log; large parts are reduced to sha256 digests."""
    assert log.status == "success", (log.eval.task, log.status, log.error)
    samples = sorted(log.samples or [], key=lambda s: (str(s.id), s.epoch))
    inputs, events, scores = [], [], []
    for sample in samples:
        inputs.append([str(sample.id), sample.epoch, _input(sample.input), sample.target,
                       sample.metadata])
        calls = []
        for event in sample.events:
            if event.event != "model":
                continue
            calls.append(json.dumps(
                [event.role, [_message(m) for m in event.input],
                 event.config.model_dump(exclude_none=True), event.output.completion],
                ensure_ascii=False, default=str))
        events.append([str(sample.id), sample.epoch, sorted(calls)])
        scores.append([str(sample.id), sample.epoch,
                       {name: [_number(s.value), s.answer, s.explanation, _number(s.metadata)]
                        for name, s in sorted((sample.scores or {}).items())}])
    # unreduced metrics (reducer None) come as a second entry under the same scorer name
    metrics = {
        score.name + ("" if score.reducer else " (unreduced)"):
        {name: _number(metric.value) for name, metric in score.metrics.items()}
        for score in (log.results.scores if log.results else [])
    }
    return {
        "name": log.eval.task,
        "dataset_samples": log.eval.dataset.samples,
        "epochs": log.eval.config.epochs,
        "reducer": log.eval.config.epochs_reducer,
        "solvers": [step.solver for step in log.plan.steps],
        "scorers": [scorer.name for scorer in log.eval.scorers or []],
        "remote": remote_calls,
        "task_metadata": _sha(log.eval.metadata),
        "inputs": _sha(inputs),
        "model_calls": _sha(events),
        "scores": _sha(scores),
        "metrics": _sha(metrics),
        "_detail": {"inputs": inputs, "events": events, "scores": scores, "metrics": metrics,
                    "task_metadata": log.eval.metadata},
    }

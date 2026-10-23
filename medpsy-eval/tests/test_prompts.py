"""Inference templates of the closed-ended units, byte for byte against the paper templates."""

from __future__ import annotations

import importlib

import pytest

from medpsy_eval.prompting import render_messages

PAPER_TEMPLATES = {
    'afrimedqa': 'Question: {question}\n\nOptions:\n{choices}\n\nAnswer:',
    'medmcqa': '{question}\n\nAnswer:',
    'medqa': 'Question: {question}\n\nOptions:\n{choices}\n\nAnswer:',
    'medxpertqa': 'Q: {question}\nA: Among {start} through {end}, the answer is',
    'mmlu': 'Question: {input}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:',
    'mmlu_pro_health': 'Question: {question}\n\nOptions:\n{options_str}\n\nAnswer:',
    'pubmedqa': '{question}\n\nOptions:\n{choices}\n\nAnswer:',
}


@pytest.mark.parametrize("unit", PAPER_TEMPLATES)
def test_unit_templates_are_byte_identical(unit: str) -> None:
    module = importlib.import_module(f"medpsy_eval.benchmarks.{unit}")
    prompts = importlib.import_module(f"medpsy_eval.benchmarks.{unit}.prompts")
    assert module.INFER_USER == {"v1": PAPER_TEMPLATES[unit]}
    assert prompts.QUESTION_FORMAT == PAPER_TEMPLATES[unit]


def test_system_text_is_sent_as_a_system_message_without_trailing_whitespace() -> None:
    from medpsy_eval.benchmarks.medmcqa import prompts

    messages = render_messages(prompts.QUESTION_FORMAT, {"question": "Q {x}?", "x": 1},
                               prompts.INFER_SYSTEM)
    assert [(m.role, m.text) for m in messages] == [
        ("system", "You are a helpful medical assistant."), ("user", "Q 1?\n\nAnswer:")]
    assert [(m.role, m.text) for m in render_messages("{{A}} {a}", {"a": "1"})] == [
        ("user", "{{A}} 1")]

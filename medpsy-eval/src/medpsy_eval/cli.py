"""``medpsy-eval`` command line."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .presets import SuiteComposition
    from .suites import Suite


@dataclass
class EvalPlan:
    """What ``eval`` evaluates: one suite factory or benchmark unit, its variants and the task
    arguments."""

    suite: Suite
    datasets: list[str]
    preset_args: dict[str, object]
    user_args: dict[str, object]
    composition: SuiteComposition | None

    def build_tasks(self) -> list:
        args = {**self.preset_args, **self.user_args}
        if not self.suite.per_dataset:
            return [self.suite.factory(**args)]
        return [self.suite.factory(dataset=dataset, **args) for dataset in self.datasets]


def _user_args(values: list[str] | None, suite: Suite) -> dict[str, object]:
    import yaml

    out = {}
    for kv in values or []:
        key, sep, value = kv.partition("=")
        if not sep:
            raise ValueError(f"-T expects key=value, got {kv!r}")
        if key not in suite.arguments:
            hint = "; select variants with --datasets" if key == "dataset" else ""
            raise ValueError(f"{suite.name} has no task argument {key!r}; "
                             f"allowed: {list(suite.arguments)}{hint}")
        out[key] = yaml.safe_load(value)
    return out


def plan_eval(args: argparse.Namespace) -> EvalPlan:
    """Resolve preset, suite (or unit, without a preset), variants and task arguments
    (``ValueError`` if invalid)."""
    from .presets import compose_suite
    from .suites import get_suite
    from .units import get_task

    composition = compose_suite(args.preset, args.suite) if args.preset else None
    if composition:
        preset_datasets = list(composition.datasets)
        preset_args = composition.task_args
        suite = get_suite(composition.task)
    else:
        task_name = args.suite or "closed_ended_v1"
        preset_datasets, preset_args = [], {}
        suite = get_task(task_name)
    datasets = args.datasets.split(",") if args.datasets else preset_datasets
    for dataset in datasets:
        suite.check_dataset(dataset)
    if suite.per_dataset and not datasets:
        raise ValueError(f"{suite.name} needs --datasets or a preset dataset list")
    user_args = _user_args(args.T, suite)
    return EvalPlan(suite, datasets, preset_args, user_args, composition)


def grader_role(judge: str, base_url: str | None, max_connections: int | None,
                from_preset: bool) -> dict:
    """``grader`` model role. A judge named by a preset or log is a Hugging Face id, served by
    vLLM when ``base_url`` is given."""
    from inspect_ai.model import GenerateConfig, get_model

    if from_preset and base_url and not judge.startswith("openai/"):
        judge = f"vllm/{judge}"
    config = GenerateConfig(max_connections=max_connections)
    return {"grader": get_model(judge, base_url=base_url, config=config)}


def print_metrics(log) -> None:
    metrics = log.results.scores[0].metrics if log.results and log.results.scores else {}
    print(log.eval.task, log.status, {k: round(v.value, 6) for k, v in metrics.items()})


def cmd_eval(args: argparse.Namespace) -> None:
    from inspect_ai import eval as inspect_eval

    try:
        plan = plan_eval(args)
    except ValueError as e:
        raise SystemExit(f"medpsy-eval eval: {e}") from None
    suite, composition = plan.suite, plan.composition
    judge_name = args.judge
    if judge_name and suite.judge is None:
        raise SystemExit(f"medpsy-eval eval: {suite.name} has no judge; drop --judge")
    try:
        tasks = plan.build_tasks()
    except ValueError as e:
        raise SystemExit(f"medpsy-eval eval: {e}") from None

    generation = composition.generation if composition else {}
    judge_spec = composition.judge if composition else None
    judge_meta = {"model": judge_name} if judge_name else None
    if judge_name is None and judge_spec and suite.judge:
        judge_meta = dict(judge_spec)
    model_roles = None
    if judge_meta and not args.no_score:
        model_roles = grader_role(judge_meta["model"], args.judge_base_url,
                                  args.judge_max_connections, from_preset=args.judge is None)
    extra_body = dict(generation.get("extra_body", {}))
    # Inspect's vllm provider never sends GenerateConfig.top_k
    top_k = args.top_k if args.top_k is not None else generation.get("top_k")
    if top_k is not None:
        extra_body["top_k"] = top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    logs = inspect_eval(
        tasks,
        model=args.model,
        model_base_url=args.base_url,
        model_roles=model_roles,
        log_dir=args.log_dir,
        limit=args.limit,
        max_samples=args.max_samples,
        max_connections=args.max_connections,
        max_tokens=args.max_tokens if args.max_tokens is not None else generation.get("max_tokens"),
        temperature=(args.temperature if args.temperature is not None
                     else generation.get("temperature")),
        top_p=args.top_p if args.top_p is not None else generation.get("top_p"),
        extra_body=extra_body or None,
        metadata={"judge": judge_meta} if judge_meta else None,
        score=not args.no_score,
        log_realtime=args.log_realtime,
        log_buffer=args.log_buffer,
        display=args.display,
    )
    for log in logs:
        print_metrics(log)


def scored_path(log_path: str, log_dir: str | None, judge: str) -> Path:
    """``<log stem>_scored_<judge>.eval`` next to the input log, or in ``log_dir``."""
    source = Path(log_path)
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", judge)
    return Path(log_dir or source.parent) / f"{source.stem}_scored_{slug}.eval"


def cmd_score(args: argparse.Namespace) -> None:
    """Score saved logs with a judge. Scorers come from rebuilding each log's task from its
    recorded registry name and arguments: Inspect logs scorer names without their package
    namespace and do not round-trip every scorer argument (HealthBench's subset id sets)."""
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log

    from .units import get_task

    for path in args.logs:
        log = read_eval_log(path)
        name = (log.eval.task_registry_name or "").removeprefix("medpsy_eval/")
        try:
            suite = get_task(name)
        except ValueError as e:
            raise SystemExit(f"medpsy-eval score: {path}: {e}") from None
        if suite.judge is None:
            raise SystemExit(f"medpsy-eval score: {path}: {suite.name} has no judge")
        recorded = (log.eval.metadata or {}).get("judge")
        judge_meta = {"model": args.judge} if args.judge else recorded
        if not judge_meta:
            raise SystemExit(f"medpsy-eval score: {path}: the log records no judge; pass --judge")
        task = suite.factory(**log.eval.task_args)
        roles = grader_role(judge_meta["model"], args.judge_base_url, args.judge_max_connections,
                            from_preset=args.judge is None)
        scored = score(log, task.scorer, model_roles=roles, action="overwrite",
                       display=args.display)
        scored.eval.metadata = {**(scored.eval.metadata or {}), "judge": judge_meta}
        out = scored_path(path, args.log_dir, judge_meta["model"])
        write_eval_log(scored, str(out))
        print_metrics(scored)
        print("log:", out)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["serve"]:
        from .serve import main as serve_main

        return serve_main(argv[1:])
    from .presets import PRESET_NAMES

    ap = argparse.ArgumentParser(prog="medpsy-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="generate and score")
    ev.add_argument("--preset", choices=PRESET_NAMES,
                    help="load benchmark selection and run settings from a paper preset")
    ev.add_argument("--suite",
                    help="preset suite name, or a suite factory or benchmark unit name when "
                         "--preset is omitted")
    ev.add_argument("--datasets", default="",
                    help="comma-separated override (defaults to the preset's benchmarks)")
    ev.add_argument("--model", default="none/none")
    ev.add_argument("--base-url")
    ev.add_argument("--judge", help="grader model, e.g. vllm/Qwen/QwQ-32B")
    ev.add_argument("--judge-base-url", help="base URL of a running judge server")
    ev.add_argument("--log-dir", default="logs")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--max-connections", type=int)
    ev.add_argument("--max-samples", type=int)
    ev.add_argument("--judge-max-connections", type=int)
    ev.add_argument("--max-tokens", type=int)
    ev.add_argument("--temperature", type=float)
    ev.add_argument("--top-p", type=float)
    ev.add_argument("--top-k", type=int)
    ev.add_argument("--min-p", type=float)
    ev.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    ev.add_argument("--no-score", action="store_true", help="generate only (infer)")
    ev.add_argument("--log-realtime", action="store_true",
                    help="stream sample events to Inspect's realtime buffer (slow on NFS)")
    ev.add_argument("--log-buffer", type=int, default=500)
    ev.add_argument("--display", default=None, help="Inspect display: full, rich, plain, none")
    ev.add_argument("-T", action="append",
                    help="task arg key=value (a parameter of the suite factory or unit)")
    ev.set_defaults(func=cmd_eval)

    sc = sub.add_parser("score", help="score saved logs (e.g. from eval --no-score) with a judge")
    sc.add_argument("logs", nargs="+", help="Inspect .eval log files")
    sc.add_argument("--judge", help="grader model, e.g. vllm/Qwen/QwQ-32B "
                                    "(default: the judge recorded in the log)")
    sc.add_argument("--judge-base-url", help="base URL of a running judge server")
    sc.add_argument("--judge-max-connections", type=int)
    sc.add_argument("--log-dir", help="where to write scored logs (default: next to each input)")
    sc.add_argument("--display", default=None, help="Inspect display: full, rich, plain, none")
    sc.set_defaults(func=cmd_score)

    sub.add_parser("serve", help="start vllm serve in the background and wait until it answers "
                                 "(same arguments as vllm serve)")
    args = ap.parse_args(argv)
    if getattr(args, "preset", None) and not args.suite:
        ap.error("--preset requires --suite")
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

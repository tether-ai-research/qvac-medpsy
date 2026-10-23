# medpsy-eval

Evaluation pipeline for the MedPsy model releases, built on
[Inspect](https://inspect.aisi.org.uk/). It reproduces the closed-ended and HealthBench
results reported for [MedPsy-4B](https://huggingface.co/qvac/MedPsy-4B) and
[MedPsy-1.7B](https://huggingface.co/qvac/MedPsy-1.7B), and evaluates any model served behind
an OpenAI-compatible endpoint with the same protocol.

## What it runs

The `medpsy` preset ([`configs/presets/medpsy.yaml`](src/medpsy_eval/configs/presets/medpsy.yaml))
defines two suites and pins the generation settings, judge models and judge revisions.

| Suite | Benchmarks | Epochs | Judge | Metric |
| ----- | ---------- | -----: | ----- | ------ |
| `closed_ended` | `mmlu` (6 health subjects), `mmlu_pro_health`, `medmcqa`, `MedQA_US`, `medxpertqa`, `AfriMedQA`, `PubMedQA` | 3 | `Qwen/QwQ-32B` | accuracy ↑ |
| `healthbench` | HealthBench Standard, Hard and Consensus, with per-axis and per-theme scores | 1 | `opencompass/CompassJudger-2-32B-Instruct` | rubric score ↑ |

Closed-ended answers are generated free-form. The judge reads the answer (with the reasoning
trace removed) and extracts the chosen option; no-answer and multiple-answer extractions count
as wrong. HealthBench sends one grading request per rubric item and reports the clipped mean
of achieved over possible points.

Model generation uses `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`, thinking
enabled, and up to 16,384 output tokens. Datasets are downloaded from the Hugging Face Hub at
pinned revisions.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra serve   # includes vLLM; plain `uv sync` if you bring your own servers
```

## Run

`medpsy-eval eval` talks to OpenAI-compatible servers: one for the evaluated model and one for
the judge of the suite you run.

Hardware: the 4B or 1.7B model fits on one GPU; each 32B judge needs two 80 GB GPUs in
bfloat16 (raise `--tensor-parallel-size` if you have more). With the layout below, the model
and both judges fit on five GPUs and the two suites can run at the same time.

### 1. Start the servers

Pick one option. `medpsy-eval serve` takes the same arguments as `vllm serve`, starts the
server in the background and returns once it answers; it prints the command that stops it
and writes vLLM's output to `vllm-<port>.log`. Start only the judge of the suite you run.

```bash
# --- Option A: medpsy-eval serve (waits until each server answers) ---
CUDA_VISIBLE_DEVICES=0 uv run medpsy-eval serve qvac/MedPsy-4B --port 8000 \
  --reasoning-parser qwen3 --max-model-len 32768
CUDA_VISIBLE_DEVICES=1,2 uv run medpsy-eval serve Qwen/QwQ-32B --port 8001 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --reasoning-parser deepseek_r1 --max-model-len 32768 --tensor-parallel-size 2
CUDA_VISIBLE_DEVICES=3,4 uv run medpsy-eval serve opencompass/CompassJudger-2-32B-Instruct --port 8002 \
  --revision 1e586e14ba945e073fb60d73e91ce798e323341c \
  --max-model-len 32768 --tensor-parallel-size 2

# --- Option B: plain vllm serve in the background, then wait by hand ---
CUDA_VISIBLE_DEVICES=0 uv run vllm serve qvac/MedPsy-4B --port 8000 \
  --reasoning-parser qwen3 --max-model-len 32768 &
CUDA_VISIBLE_DEVICES=1,2 uv run vllm serve Qwen/QwQ-32B --port 8001 \
  --revision 976055f8c83f394f35dbd3ab09a285a984907bd0 \
  --reasoning-parser deepseek_r1 --max-model-len 32768 --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=3,4 uv run vllm serve opencompass/CompassJudger-2-32B-Instruct --port 8002 \
  --revision 1e586e14ba945e073fb60d73e91ce798e323341c \
  --max-model-len 32768 --tensor-parallel-size 2 &
for port in 8000 8001 8002; do   # the ports you started
  until curl -sf localhost:$port/v1/models > /dev/null; do sleep 10; done
done
```

### 2. Run the suites

```bash
# Closed-ended benchmarks, judged by QwQ-32B (port 8001)
uv run medpsy-eval eval --preset medpsy --suite closed_ended \
  --model vllm/qvac/MedPsy-4B --base-url http://localhost:8000/v1 \
  --judge-base-url http://localhost:8001/v1 --log-dir logs/closed_ended

# HealthBench, judged by CompassJudger-2 (port 8002)
uv run medpsy-eval eval --preset medpsy --suite healthbench \
  --model vllm/qvac/MedPsy-4B --base-url http://localhost:8000/v1 \
  --judge-base-url http://localhost:8002/v1 --log-dir logs/healthbench
```

The judge's reasoning parser matters: QwQ-32B is served with `--reasoning-parser deepseek_r1`
so only its final answer is parsed; CompassJudger-2 runs without one.

### Generate once, score with several judges

`--no-score` saves the model's generations without judging them, and no judge server is
needed. `medpsy-eval score` then grades saved logs, writing one new log per judge
(`<log>_scored_<judge>.eval`, next to the input or in `--log-dir`) and leaving the input
untouched:

```bash
uv run medpsy-eval eval --preset medpsy --suite closed_ended --no-score \
  --model vllm/qvac/MedPsy-4B --base-url http://localhost:8000/v1 \
  --log-dir logs/generations

# the judge recorded in the log (the preset's QwQ-32B)
uv run medpsy-eval score logs/generations/*.eval --judge-base-url http://localhost:8001/v1
# a second judge on the same generations
uv run medpsy-eval score logs/generations/*.eval \
  --judge vllm/<judge-model> --judge-base-url http://localhost:8003/v1
```

The judge's temperature and token limit come from the task arguments in the log. Use
`medpsy-eval score` rather than `inspect score`: it rebuilds each task's scorers from the
log, which `inspect score` cannot do for these tasks.

### Resuming a run

A task that errored or was interrupted resumes with `uv run inspect eval-retry <log>`:
completed samples are kept and only the missing ones are generated and scored. The model and
judge servers must be reachable at the same URLs as in the original run.

### Useful options

| Option | Effect |
| ------ | ------ |
| `--datasets medxpertqa,PubMedQA` | Run a subset of the suite's benchmarks |
| `--limit 20` | Evaluate only the first N samples per benchmark (smoke test) |
| `-T epochs=1` | Override a suite argument from the preset |
| `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--[no-]enable-thinking` | Override generation settings |
| `--judge vllm/<model>` | Grade with a different judge |
| `--no-score` | Generate only; score later with `medpsy-eval score` |
| `--max-connections`, `--judge-max-connections`, `--max-samples` | Concurrency |

To evaluate another model, serve it and pass `--model vllm/<served-name>`; drop
`--reasoning-parser` for non-reasoning models.

## Outputs

Each task writes an Inspect log (`.eval`) to `--log-dir` and prints its headline metrics.
Browse logs with `uv run inspect view --log-dir logs`, or load them with
`inspect_ai.analysis.evals_df` / `samples_df`.

MMLU Health is the `mmlu` task's `subject_mean` metric, the unweighted mean of its six
subjects (its `accuracy` pools all questions instead). The closed-ended Average is the
unweighted mean of the seven benchmarks.
Besides accuracy, the logs carry no-answer and multiple-answer rates and per-sample token
counts for the reasoning, the answer and the total.

### Reading results

Metrics that count every epoch (`no_answer_rate`, `multiple_answers_rate`, the HealthBench axis/theme scores) are written to the log as a second `results.scores` entry with the same scorer name and `reducer: null`. Metric names never collide, so merge entries by name, or key them by `(name, reducer)`, rather than keeping one entry per name.

## Reproduction check

A fresh run of both suites on `qvac/MedPsy-4B` (BF16, vLLM, `--max-tokens 8192`) against the
reported numbers:

| Benchmark | Reported | Reproduced |
| --------- | -------: | ---------: |
| Closed-ended Average | 70.54 | 70.39 |
| MMLU Health | 89.70 | 89.21 |
| AfriMedQA | 71.50 | 71.32 |
| MMLU-Pro Health | 70.45 | 70.42 |
| MedMCQA | 72.15 | 72.15 |
| MedQA (USMLE) | 84.39 | 84.76 |
| MedXpertQA | 30.61 | 30.86 |
| PubMedQA | 75.00 | 74.00 |
| HealthBench | 74.00 | 73.42 |
| HealthBench Hard | 58.00 | 57.99 |

Differences are within run-to-run sampling variation; small benchmarks such as PubMedQA (500
questions) and the MMLU subjects move by a point or more between epochs.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```

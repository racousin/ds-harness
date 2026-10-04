# DS-Harness starter kit

A small language model cannot compute a variance over 800 rows, or fit a classifier, in its
head. It can read a question and write ten lines of pandas. Your job is the **harness**: the
program around the model that routes each task, prompts it, runs its code, checks the result
and recovers from failures.

## Files

| File | What it is |
|---|---|
| `dsh.py` | The kit: one module, upload it next to your `agent.py` |
| `agent_naive.py` | Grading anchor A0: the model answers directly, no tools |
| `agent_kit_baseline.py` | Grading anchor A1: one code attempt, one repair, fallbacks |
| `local_eval.py` | Runs any agent file on a task file and scores it with the challenge scorer |
| `test_dsh.py` | Unit tests of `dsh` (no model needed): `pytest test_dsh.py` |

## What the kit gives you, and what you build

`dsh` provides the plumbing that is fiddly or unsafe to get right:

- `load_llm(model_id)` returns an object with `.chat(conversations, max_new_tokens, temperature,
  max_time=...)`. It generates in batches with left padding, uses fp16 on GPU and fp32 on CPU,
  and turns Qwen3 "thinking" off unless you ask for it. `max_time` stops generation when the
  time runs out: a `generate()` call cannot be interrupted any other way.
- `load_llm` pins `revision=` to the commit of the cached snapshot (`cached_revision`): the
  platform's upload scan (bandit rule B615) refuses a `from_pretrained` call without one. It
  sets `trust_remote_code=True` for the four cached models that need it (`REMOTE_CODE_MODELS`).
- `run_python(code, files, timeout_s)` runs model-written code in a subprocess, in a fresh
  directory under `/tmp` that holds the task's CSVs. It has a timeout, a memory cap on Linux and
  cleans up afterwards. It returns `ExecResult(ok, value, stdout, error, seconds)`, where `value`
  is the variable **`RESULT`**. Only `RESULT` counts. Accepting `result = ...` or a final bare
  expression is a design choice left to you.
- `calculator(expr)` is a safe arithmetic evaluator.
- `csv_overview(files)` describes each file: its shape, dtypes, missing values and first rows.
  Small files (a `data_dictionary.csv`, a lookup table) are shown in full, exactly as written.
- `coerce_number`, `coerce_label`, `coerce_list` and `to_answer` apply the **scorer's own**
  coercion rules (`scoring.py`).
  `parse_number`, `parse_label` and `parse_list` pull an answer out of free text.
- `Budget.for_tasks(tasks)` is a stopwatch against the tasks' `time_budget_s`.

You build everything else, and the oral asks you to justify it: routing, prompts and few-shot
banks, the tools you expose to the model (e.g. `fit_and_predict`, `forecast`), the tool-call
format, retries, verification, self-consistency, the choice of model, and any fine-tuning.

## The agent interface

```python
class Agent:
    def __init__(self):            # load the model here, in less than 60 s
        ...
    def solve(self, tasks):        # tasks: list of {"id", "prompt", "files", "answer_type", "time_budget_s"}
        return [{"id": ..., "answer": ..., "trace": "..."} for t in tasks]
```

`files` maps file names to CSV text. `answer_type` is one of `number`, `category`, `list`
(labels in order), `vector` (a forecast) or `predictions` (one value per *requested* test row:
the prompt may ask for a subset of rows, for probabilities, or for another unit).
`schema.md`, next to `dev.json`, is the reference. Level-3 files hold traps the prompt tells
you about: columns you must not use (recorded after the outcome, or leaking it) and missing
values in the features.

### Time

Each task's `time_budget_s` is its share of the call's timeout (the formula and the delivery
are in `schema.md`). The job ends 399 s after it starts, model loading included, and a batch
whose full timeout no longer fits before the end is not sent: its tasks score 0. **Plan for
330 s of answering**: about 2 s per level-1/2 task and 8 s per level-3 task. The kit
baseline already uses most of that time: adding a second repair round to it lowered its score,
because tasks went unanswered.

### A failed call means no score

If `solve` raises an exception or misses its timeout, the run ends and the platform fails the
deployment: **no score**, and the deployment still counts in your quota. So a bug that affects
one task must not escape `solve`. Both agents in the kit catch every failure **per task** and
answer that task with a well-typed placeholder. Stopping on time is the same rule: check
`Budget.remaining()` before each expensive step, and pass `max_time` to `chat`. Memory too:
going over the container's 3 GiB of RAM kills the agent, so plan for it as a failure; a
`torch.cuda.OutOfMemoryError` is an ordinary exception, catch it per task.

### Submitting

Upload `agent.py` (your agent, or a baseline renamed) and the modules it imports, such as
`dsh.py`: up to 10 files and 100 MB. They land in one read-only directory on `sys.path`.
Choose the **PyTorch** runtime: the console's default is not torch, and a wrong pick costs a
deployment. You have **2 deployments per person per rolling 24 h, counted across every
ML-Arena challenge, failed ones included**. Each one is a short test run on a few dev tasks,
then the scored run on the private set: about 7–9 min, plus the queue (one GPU, one job at a
time).

## Running locally

With [uv](https://docs.astral.sh/uv/):

```bash
uv run --with numpy --with pandas --with pytest  pytest test_dsh.py
uv run --with torch --with transformers --with accelerate --with pandas --with numpy \
       python local_eval.py agent_kit_baseline.py --limit 20
```

With pip: `pip install -r requirements.txt`, then run the same `python ...` commands.

`local_eval.py` reads the `dev.json` next to it (use `--dev` to point elsewhere) and delivers
the tasks through `run_agent` from the `scoring.py` next to it. That is the platform's own
code, and `localtest.py` uses it too, so batches, timeouts, `time_budget_s`, scoring and "one
failed call ends the run" all behave as on the platform. On top of that, `local_eval.py`:
- times `Agent()` against the 60 s limit,
- prints each task's score and its call's duration against the timeout,
- prints the traceback when a call fails,
- saves the full traces with `--out run.json`. Your error analysis starts there.

`--limit N` takes N tasks (N >= 3) spread over the three levels. `--budget-s 330` adds the
private run's total budget, scaled to the tasks you run: the 330 s are for the 119-task private
mix, and the full `dev.json` (178 tasks, more of them level 3) gets about 575 s. `--no-scale`
uses the number as given. Expect dev scores to run higher than private ones.

To try a smaller model without editing the file, set `DSH_MODEL`, e.g.
`DSH_MODEL=Qwen/Qwen2.5-0.5B-Instruct python local_eval.py agent_kit_baseline.py`.

**Match the platform.** Its packages are torch, transformers, accelerate, pandas, numpy, sympy
and matplotlib. It does **not** have scikit-learn, scipy or statsmodels. Colab has them, so code
that imports them runs on Colab and then fails on the platform. Tell your model that they are
unavailable, and test for it.

### Colab (free T4, 15 GB)

Colab already has torch, transformers and pandas (and scikit-learn, which the platform does
not have). Upload the kit and the dev files, then:

```python
!pip -q install accelerate
!python local_eval.py agent_kit_baseline.py --limit 30
```

A 1.5B model in fp16 plus batch-8 generation fits easily. A 3B model fits as well.

### CPU only

It works, but it is slow. Without a GPU, `load_llm` uses fp32, and generating a few hundred tokens
per task with a 1.5B model takes tens of seconds to minutes per task on a laptop (not
measured on every machine: time a 5-task run first). For development on
CPU, use `Qwen/Qwen2.5-0.5B-Instruct` and `--limit 10`, and move to Colab for full runs.
`run_python` itself is cheap on any machine: about 0.5–1 s per call, mostly the pandas import.

## Memory per model size

| Model size | fp16 weights (GPU) | fp32 weights (CPU) | Example |
|---|---|---|---|
| 0.5B | ~1.0 GB | ~2.0 GB | Qwen2.5-0.5B-Instruct |
| 1–1.5B | ~2.5–3.1 GB | ~5–6 GB | Qwen2.5-1.5B-Instruct, Falcon3-1B |
| 1.7–2B | ~3.5–4.1 GB | ~7–8 GB | Qwen3-1.7B, SmolLM2-1.7B |
| 2.4–3B | ~5–6.2 GB | ~10.5–12.5 GB | EXAONE-3.5-2.4B, SmolLM3-3B |

Add 0.5–2 GB for activations and the KV cache, depending on batch size and prompt length. On
the platform the GPU has 24 GB, but the **agent container has only 3 GiB of RAM**. `load_llm`
therefore streams the weights straight to the GPU. Do not build a second copy of the model on
the CPU. `run_python` children count toward those 3 GiB too, and each is capped at 1 GiB by
default. Never `import torch` inside sandboxed code.

## Platform facts worth knowing

- There is no network. Models come from the platform's read-only Hugging Face cache. These
  load with `dsh.load_llm`: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct,
  Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2-1.5B-Instruct, Qwen/Qwen3-1.7B,
  HuggingFaceTB/SmolLM2-1.7B-Instruct, HuggingFaceTB/SmolLM3-3B,
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, allenai/OLMo-2-0425-1B-Instruct,
  stabilityai/stablelm-2-1_6b-chat, tiiuae/Falcon3-1B-Instruct,
  TinyLlama/TinyLlama-1.1B-Chat-v1.0, and, with `trust_remote_code` (set for you),
  IndexTeam/Index-1.9B-Chat, LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct,
  internlm/internlm2_5-1_8b-chat, openbmb/MiniCPM-2B-sft-bf16.
- The root filesystem is read-only. `/tmp` is the only writable place, a 128 MB tmpfs.
  `run_python` writes there and cleans up.
- The agent has 3 CPUs, 3 GiB of RAM and one 24 GB GPU (RTX 4090). `__init__` must finish
  in 60 s.
- One failed or late call means no score. Skipping a retry costs one task.
- Do not log or store task content (prompts, files) from platform runs: the private set stays
  private.

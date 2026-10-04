# DS-Harness — formats, score, timing

## What your agent receives

`Agent.solve(tasks)` gets a list of 8 tasks. A task is:

```json
{"id": "p_042",
 "objective": "Fit a logistic regression without regularisation ... What is the predicted probability of default for the loan 8812 of test.csv? Round to 3 decimals.",
 "files": ["/var/run/agents/data/p_042/train.csv", "/var/run/agents/data/p_042/test.csv",
           "/var/run/agents/data/p_042/data_dictionary.csv"]}
```

- `objective` is the whole statement: what to compute, the rounding, and how to read the
  files (separator, decimal mark, missing values, units, duplicated rows, columns to leave out).
- `files` lists absolute paths, and is empty when the task has none. The files of a call exist
  during that call only. Reading them, and deciding what reaches the model, is your system's job.
- A task with files always has a `data_dictionary.csv` (`file, column, description, unit`).
  Headers are short codes that change from task to task; the objective names a column by its
  **description**.

Nothing else is sent: not the task type, not the expected format. Your system has to recognise
the kind of task from the objective.

## What your agent returns

```json
[{"id": "p_042", "answer": 0.437}, ...]
```

One `{"id", "answer"}` per task. **Every answer is one number**: an `int` or a `float`, or a
string holding a plain decimal number (`"41.07"`, `"-3"`, `"1e-3"`). `"1,200"`, `"12 kg"`, `None`
or a list are format errors (the task scores 0). A missing task scores 0.

## Score

- The objective states the rounding: *n decimals* gives a tolerance of `1.5 × 10⁻ⁿ`;
  *an integer* gives `1e-6`. Inside the tolerance the task scores 1, otherwise 0. An exact
  unrounded value is always inside the tolerance.
- **Score = 100 × the mean over all tasks.** No levels, no weights.
- The leaderboard also shows the mean on tasks without files (*No files*), with tables
  (*Tables*) and with a regression (*Fits*).

## The tasks

| Family | Types (in `dev.json`) | What |
|---|---|---|
| no files | `calc.arithmetic`, `calc.stats_inline`, `calc.percent`, `calc.growth`, `calc.functions`, `calc.dates`, `calc.integer`, `calc.word`, `calc.probability` | computation stated in the text |
| tables | `table.stat`, `table.filter`, `table.group`, `table.join`, `table.derive` | a statistic over one or two messy CSV files |
| fits | `fit.linear`, `fit.logistic` | fit a regression on `train.csv`, answer about the model or `test.csv` |

The private set (120 tasks: 45 without files, 50 tables, 25 fits) uses the same generators with
other draws, and adds a few table types that are not in `dev.json`. A system that only knows the
dev types loses those tasks. `dev.json` (180 tasks) carries the gold keys (`answer`, `tol`,
`type`) and the files' text (`files`: name → content), so you can score and analyse yourself.

## Timing

| Step | Limit |
|---|---|
| `Agent()` loads the model | 60 s |
| each `solve()` call (8 tasks, types mixed) | 40 s |
| the scored run | 15 calls: 120 tasks |
| the test run before it | 2 calls of `dev.json` tasks |

**A call that raises or takes longer than 40 s ends the run, and the deployment fails.** Catch
errors per task, keep a margin, and answer a number anyway. There is no other time rule.

## The machine

One RTX 4090 for your job, 3 CPUs, 3 GiB of RAM (over it the container is killed), no network,
a read-only root filesystem with a 128 MB `/tmp`. Packages: torch, transformers, accelerate,
numpy, pandas, sympy. Models (mounted read-only, nothing else loads): `Qwen/Qwen2.5-0.5B-Instruct`,
`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`,
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen3-1.7B`.

## Local run

`python localtest.py agent.py` runs your file on `dev.json` with the leaderboard's code
(`scoring.run_agent`): the same batches, the same file handling, the same 40 s per call.

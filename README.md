# DS-Harness — evaluation code and starter kit

The code of **DS-Harness**, the project of MS2A — Machine Learning Practice on ML-Arena
(challenge 194, [ml-arena.com/viewchallenge/194](https://ml-arena.com/viewchallenge/194)).
You build a **harness**: the program around a small LLM that answers data-science tasks
(prompts, routing, code execution, tools, checks, model choice). This repository holds the
scorer the leaderboard runs, local runners and a starter kit. It holds no data: `dev.json`
(178 public tasks with answers) comes from the challenge page.

Score = 100 × (0.3·L1 + 0.4·L2 + 0.3·L3). The private set has 119 tasks (37 level 1,
67 level 2, 15 level 3), including families absent from `dev.json`. The kit baseline
(`agent_kit_baseline.py`) scores 13.9 on it (L1 / L2 / L3: 32.4 / 10.4 / 0; 103 of 119 tasks answered before time ran out). Formats, scoring
and delivery: [`schema.md`](schema.md).

## Quick start

```bash
git clone https://github.com/racousin/ds-harness && cd ds-harness
pip install -r requirements.txt mlarena-sdk
python -c "import mlarena; mlarena.connect('mlk_user_...').download_dataset(194, '.')"
python local_eval.py agent_kit_baseline.py --limit 30       # 30 tasks, scores and traces (GPU)
python localtest.py agent_kit_baseline.py --budget-s 330    # all dev tasks, the platform's rules
```

Your API key (`mlk_user_...`) is on your ML-Arena profile page. `download_dataset` writes
`dev.json` next to the code (it is gitignored). No GPU? The starter notebook runs all of this on
a free Colab T4, up to a submission:
[open it in Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-project-ds-harness.ipynb).

`--budget-s 330` is the budget of the 119-task private mix; the runners scale it to the tasks
you run (about 575 s for the full `dev.json`). Expect dev scores to run higher than private
ones.

## Files

| File | What it is |
|---|---|
| `scoring.py` | the scorer and the delivery loop (`run_agent`) the leaderboard runs |
| `env.py` | the platform environment that calls it on the private set |
| `schema.md` | task and answer formats, scoring, delivery and time |
| `localtest.py` | runs an agent on a task file with the platform's rules |
| `local_eval.py` | the same, plus timings per call, full traces (`--out`) and `--slowdown` |
| `dsh.py` | the kit: model loading, code sandbox, calculator, parsing, time budget |
| `agent_naive.py` | grading anchor A0: the model answers directly |
| `agent_kit_baseline.py` | grading anchor A1: the model writes Python, the kit runs it, one repair |
| `kit_README.md` | the kit's documentation: platform facts, models, memory, Colab |
| `test_dsh.py` | unit tests of the kit (no model needed) |

`scoring.py` and `env.py` are what the leaderboard runs. Do not edit them: a change only makes
your local numbers wrong.

## Submitting

Your submission is `agent.py` (it defines `class Agent`) plus the modules it imports, e.g.
`dsh.py`: up to 10 files, 100 MB.

```python
class Agent:
    def __init__(self):        # load the model: 60 s at most
        ...
    def solve(self, tasks):    # tasks: list of {"id", "prompt", "files", "answer_type", "time_budget_s"}
        return [{"id": t["id"], "answer": ..., "trace": "..."} for t in tasks]
```

- **A failed call means no score.** If `solve` raises or misses its timeout, the run ends and
  the deployment fails. Wrap each task in try/except, return a placeholder answer instead of
  raising, and stop at the task's `time_budget_s`.
- **Time.** The job ends 399 s after it starts, model loading included. A batch whose full
  timeout no longer fits is not sent (its tasks score 0). Plan for 330 s of answering: about
  2 s per level-1/2 task and 8 s per level-3 task.
- **Platform.** One 24 GB GPU, 3 CPUs, 3 GiB RAM, 128 MB of `/tmp`, no network. Models come
  from the platform's offline cache (list in `kit_README.md`), loaded with `dsh.load_llm`.
  Packages: torch, transformers, accelerate, pandas, numpy, sympy, matplotlib. **No
  scikit-learn, scipy or statsmodels.**
- **Runtime.** Choose the **PyTorch** runtime: the console's default is not torch.
- **Quota.** 2 deployments per person per rolling 24 h, counted across every ML-Arena
  challenge, failed ones included. A deployment takes about 7–9 min plus the queue.
- **Rules.** Work in pairs. Do not log or store task content (prompts, files) from platform
  runs.

From Python (the file must be named `agent.py`):

```python
import mlarena
client = mlarena.connect("mlk_user_...")
client.submit(194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"},
              wait=True, timeout_sec=1800)
```

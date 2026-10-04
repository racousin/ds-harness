# DS-Harness

The code of the ML-Arena challenge [DS-Harness](https://ml-arena.com/viewchallenge/194), the
project of MS2A — Machine Learning Practice: build an AI system around a small language model
that answers data-science objectives with one number each.

| file | what it is |
|---|---|
| `scoring.py` | the scorer and the delivery loop (`run_agent`) the leaderboard runs — do not edit |
| `env.py` | the platform env that calls it |
| `schema.md` | what the agent receives and returns, the score, the timing, the machine |
| `localtest.py` | runs an agent file on `dev.json` the way the leaderboard does |
| `dsh.py` | the kit: model loading, answer parsers, calculator, tool-call parser, code runner, clock |
| `stage1_direct.py` | the first solution: the model, a structured answer, a parser |
| `stage2_tool_loop.py` | a calculator called in a loop — the loop body is yours to write |
| `test_dsh.py` | tests of the kit, no model needed |
| `hf_models.json` | the five models the platform mounts for your agent |

`dev.json` (180 tasks, with their files, answers and types) is the challenge's dataset:

```python
import mlarena
client = mlarena.connect(api_key="mlk_user_...")
client.download_dataset(194, ".")
```

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q test_dsh.py
python localtest.py stage1_direct.py --test-run        # the platform's test run: 16 tasks
python localtest.py stage1_direct.py                   # all of dev.json
python localtest.py my_agent.py --type table.join -v --out run.json
```

`localtest.py` gives each call the platform's 40 s; on a slower GPU than an RTX 4090 (a Colab T4
is about three times slower) pass `--timeout 120` to measure what your system can do, then check
the time on the platform.

## Measured on dev.json

RTX 4090, the platform's agent image, the full `dev.json`:

| System | Model | Score | No files | Tables | Fits |
|---|---|---|---|---|---|
| stage 1: answer marked `ANSWER:`, parsed with a fallback | Qwen2.5-1.5B | **10.6** | 26.2 | 2.5 | 0.0 |
| stage 1 | Qwen3-1.7B | **16.7** | 44.6 | 0.0 | 2.9 |
| stage 2: + a calculator in a loop | Qwen2.5-1.5B | **13.3** | 35.4 | 1.2 | 0.0 |
| stage 2 | Qwen3-1.7B | **22.2** | 60.0 | 1.2 | 0.0 |
| stage 2 with routing: the loop without files, stage 1 with files | Qwen2.5-1.5B | **13.3** | 35.4 | 1.2 | 0.0 |
| reference: files read by the system, data view, code, one repair | Qwen2.5-1.5B | **46.1** | 86.2 | 28.8 | 11.4 |
| reference | Qwen2.5-Coder-1.5B | **53.3** | 81.5 | 47.5 | 14.3 |
| reference | Qwen3-1.7B | **62.2** | 80.0 | 62.5 | 28.6 |

## Submit

`agent.py` (your file, renamed) and the modules it imports, PyTorch runtime:

```python
client.submit(challenge_id=194, files=["agent.py", "dsh.py"],
              runtime={"language": "python", "framework": "torch"}, wait=True, timeout_sec=1800)
```

The starter notebook: [Colab](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/ms2a-machine-learning-practice/challenges/mlp-project-ds-harness.ipynb).

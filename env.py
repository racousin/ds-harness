"""DS-Harness — the MS2A-MLP project challenge (flex_v1).

Env(is_evaluation) loads private.json (score runs) or dev.json (test runs: the
first DRY_RUN_PER_FAMILY tasks of each family, so a test run stays short). Both
files and scoring.py are uploaded next to this file. evaluate(agents,
agent_infos) delivers the tasks with scoring.run_agent:

    agent.solve(tasks: list[dict]) -> list[dict]
        input  {"id", "prompt", "files", "answer_type", "time_budget_s"}
        output {"id", "answer", "trace"}

Deadline: the executor sets self.eval_context after __init__; its
"deadline_monotonic" bounds the run. Without it (local runs) the budget is
TOTAL_BUDGET_S from the start of evaluate. A call that fails or misses its
timeout ends that agent's run (the platform latches the channel): the env
returns the partial score, and the platform fails the deployment.

Metrics (the challenge's metrics_schema): score, level_1, level_2, level_3,
heldout (0-100), format_errors, tasks_answered.
"""
import json
import os
import time

from scoring import METRIC_KEYS, run_agent, validate_task

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_PATH = os.path.join(HERE, "dev.json")
PRIVATE_PATH = os.path.join(HERE, "private.json")
TOTAL_BUDGET_S = 330.0
DRY_RUN_PER_FAMILY = 2
HF_CACHE = "/opt/hf_cache"


def load_tasks(path):
    with open(path) as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path} must hold a non-empty JSON list of tasks")
    for t in tasks:
        validate_task(t)
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate task ids")
    return tasks


def dry_run_subset(tasks, per_family):
    kept, seen = [], {}
    for t in sorted(tasks, key=lambda t: t["id"]):
        if seen.get(t["family"], 0) < per_family:
            kept.append(t)
            seen[t["family"]] = seen.get(t["family"], 0) + 1
    return kept


def cached_models():
    """The models of the platform's offline HF cache, as `org/name` ids."""
    if not os.path.isdir(HF_CACHE):
        return []
    return sorted(d[len("models--"):].replace("--", "/", 1)
                  for d in os.listdir(HF_CACHE) if d.startswith("models--"))


class Env:
    def __init__(self, is_evaluation=False):
        tasks = load_tasks(PRIVATE_PATH if is_evaluation else DEV_PATH)
        self.tasks = tasks if is_evaluation else dry_run_subset(tasks, DRY_RUN_PER_FAMILY)
        print(f"{'private' if is_evaluation else 'dev'} set: {len(self.tasks)} tasks")
        print(f"models in the offline cache: {cached_models()}")

    def _deadline(self):
        ctx = getattr(self, "eval_context", None)
        if ctx is None:
            return time.monotonic() + TOTAL_BUDGET_S
        return float(ctx["deadline_monotonic"])

    def evaluate(self, agents, agent_infos):
        deadline = self._deadline()
        results = []
        for i, agent in enumerate(agents):
            t0 = time.monotonic()
            r = run_agent(agent, self.tasks, deadline)
            if set(r["metrics_detail"]) != set(METRIC_KEYS):
                raise RuntimeError(f"metric keys drifted: {sorted(r['metrics_detail'])}")
            print(f"agent {i}: {r['info_message']} | {time.monotonic() - t0:.0f} s")
            # n_episodes > 0 tells the platform the score was banked before any agent
            # failure (shared/executor/outcome_writer._run_score drops it otherwise).
            results.append({"agent_index": i, "score": r["score"],
                            "n_episodes": 1 if r["details"] else 0,
                            "metrics_detail": r["metrics_detail"],
                            "info_message": r["info_message"]})
        return {"agent_results": results}

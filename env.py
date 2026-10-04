"""DS-Harness — the MS2A-MLP project challenge (flex_v1).

Env(is_evaluation) loads private.json (the scored run: 120 tasks, 15 calls of 8)
or dev.json (the test run: 16 dev tasks, 2 calls). Both files and scoring.py are
uploaded next to this file. evaluate(agents, agent_infos) delivers the tasks with
scoring.run_agent:

    agent.solve(tasks: list[dict]) -> list[dict]
        input  {"id", "objective", "files": [absolute paths]}
        output {"id", "answer"}

The files of a batch are written under /var/run/agents/data/<id>/ (the JobPod
directory shared, read-write, by the env and agent containers) before the call
and deleted after it. Each call has CALL_TIMEOUT_S; a call that raises or runs
out of time ends the run (the platform latches the agent channel): the env
returns the partial score and the platform fails the deployment.

Metrics (the challenge's metrics_schema): score, calc, table, fit (0-100),
tasks_answered, format_errors.
"""
import json
import os
import tempfile
import time

from scoring import METRIC_KEYS, run_agent, test_run_subset, validate_task

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = "/var/run/agents"


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


class Env:
    def __init__(self, is_evaluation=False):
        if is_evaluation:
            self.tasks = load_tasks(os.path.join(HERE, "private.json"))
        else:
            self.tasks = test_run_subset(load_tasks(os.path.join(HERE, "dev.json")))
        # On the platform the shared JobPod directory exists; locally, a temp dir.
        root = SHARED if os.path.isdir(SHARED) else tempfile.mkdtemp(prefix="dsh_")
        self.data_root = os.path.join(root, "data")
        os.makedirs(self.data_root, mode=0o755, exist_ok=True)
        os.chmod(self.data_root, 0o755)
        print(f"{'private' if is_evaluation else 'dev (test run)'} set: {len(self.tasks)} tasks, "
              f"files under {self.data_root}")

    def evaluate(self, agents, agent_infos):
        results = []
        for i, agent in enumerate(agents):
            t0 = time.monotonic()
            r = run_agent(agent, self.tasks, self.data_root)
            if set(r["metrics_detail"]) != set(METRIC_KEYS):
                raise RuntimeError(f"metric keys drifted: {sorted(r['metrics_detail'])}")
            print(f"agent {i}: {r['info_message']} | {time.monotonic() - t0:.0f} s")
            # n_episodes > 0 tells the platform the score was banked before any agent
            # failure (the outcome writer drops the score of a run without episodes).
            results.append({"agent_index": i, "score": r["score"],
                            "n_episodes": 1 if r["details"] else 0,
                            "metrics_detail": r["metrics_detail"],
                            "info_message": r["info_message"]})
        return {"agent_results": results}

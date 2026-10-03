#!/usr/bin/env python3
"""Run your agent on dev.json exactly as the platform does, and print the scores.

    python localtest.py my_agent.py                    # per-call timeouts only
    python localtest.py my_agent.py --budget-s 330     # also a total budget, scaled (below)
    python localtest.py my_agent.py --out run.json     # per-task scores and traces

Your file must define `class Agent` with `solve(tasks) -> list[dict]` (see
schema.md). Delivery, timeouts and scoring come from scoring.run_agent, the
same code the platform env uses. The private set has the same format, other
wordings, other table domains and task families that are not in dev.json.

--budget-s is the budget of the private set: 119 tasks (37 level 1, 67 level 2,
15 level 3). dev.json holds more tasks and more level-3 tasks, so the budget is
scaled to the task file by the platform's per-task estimates (2 s per level-1/2
task, 8 s per level-3 task): 330 s on the private mix gives about 575 s on the
full dev.json. Pass --no-scale to use the number as given.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scoring import EST_TASK_S, LocalProxy, run_agent, validate_task  # noqa: E402

PRIVATE_MIX = {1: 37, 2: 67, 3: 15}  # tasks per level in the private set


def estimated_s(counts):
    return sum(EST_TASK_S[level] * n for level, n in counts.items())


def load_agent(path):
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_file")
    ap.add_argument("--tasks", default=os.path.join(HERE, "dev.json"))
    ap.add_argument("--budget-s", type=float,
                    help="total time budget for the 119-task private mix (plan for 330); "
                         "scaled to the task file")
    ap.add_argument("--no-scale", action="store_true", help="use --budget-s as given, unscaled")
    ap.add_argument("--family", action="append",
                    help="only these families (repeatable); together they must cover all three levels")
    ap.add_argument("--out", help="write per-task scores, errors and traces to this JSON file")
    args = ap.parse_args()
    with open(args.tasks) as f:
        tasks = json.load(f)
    for t in tasks:
        validate_task(t)
    if args.family:
        tasks = [t for t in tasks if t["family"] in set(args.family)]
    per_level = {level: sum(t["level"] == level for t in tasks) for level in PRIVATE_MIX}
    missing = [level for level, n in per_level.items() if n == 0]
    if missing:
        sys.exit(f"no task of level {missing}: the score needs all three levels")
    budget = math.inf
    if args.budget_s:
        budget = args.budget_s
        if not args.no_scale:
            budget *= estimated_s(per_level) / estimated_s(PRIVATE_MIX)
        print(f"{len(tasks)} tasks {per_level}: total budget {budget:.0f} s "
              f"({args.budget_s:.0f} s {'as given' if args.no_scale else 'for the private mix, scaled'})")
    t0 = time.monotonic()
    agent = load_agent(args.agent_file)
    print(f"agent loaded in {time.monotonic() - t0:.1f} s")
    deadline = time.monotonic() + budget
    res = run_agent(LocalProxy(agent), tasks, deadline)
    counts = {}
    for t in tasks:
        counts[t["family"]] = counts.get(t["family"], 0) + 1
    level = {t["family"]: t["level"] for t in tasks}
    print(f"{'family':30s} {'level':>5s} {'n':>4s} {'score':>7s}")
    for fam, s in sorted(res["family_scores"].items(), key=lambda kv: (level[kv[0]], kv[0])):
        print(f"{fam:30s} {level[fam]:5d} {counts[fam]:4d} {s:7.1f}")
    print(res["info_message"])
    print(f"elapsed {time.monotonic() - t0:.1f} s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()

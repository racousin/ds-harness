#!/usr/bin/env python3
"""Run your agent on dev.json the way the leaderboard runs it, and print the score.

    python localtest.py agent.py                       # all 180 dev tasks
    python localtest.py agent.py --test-run            # the platform's test run: 16 tasks, 2 calls
    python localtest.py agent.py --type calc.dates --type table.join
    python localtest.py agent.py --out run.json        # every task: answer, gold, score, error
    python localtest.py agent.py --timeout 120         # a slower machine than the platform's GPU

Your file must define `class Agent` with `solve(tasks) -> list[dict]` (schema.md).
Delivery, file handling, timeouts and scoring are scoring.run_agent, the code the
platform env runs: batches of 8, the files of a batch written to a directory
before the call and deleted after it, 40 s per call, and a call that raises or
runs out of time ends the run.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from scoring import (CALL_TIMEOUT_S, LOAD_TIMEOUT_S, LocalProxy, run_agent,  # noqa: E402
                     test_run_subset, validate_task)


def load_agent(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location("agent_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_file")
    ap.add_argument("--tasks", default=os.path.join(HERE, "dev.json"))
    ap.add_argument("--test-run", action="store_true", help="only the 16 tasks of the platform's test run")
    ap.add_argument("--type", action="append", help="only these task types (repeatable)")
    ap.add_argument("--limit", type=int, help="only the first N tasks (after the filters)")
    ap.add_argument("--timeout", type=float, default=CALL_TIMEOUT_S, help="seconds per call (platform: 40)")
    ap.add_argument("--out", help="write per-task results to this JSON file")
    ap.add_argument("-v", "--verbose", action="store_true", help="print one line per task")
    args = ap.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)
    for t in tasks:
        validate_task(t)
    if args.test_run:
        tasks = test_run_subset(tasks)
    if args.type:
        tasks = [t for t in tasks if t["type"] in set(args.type)]
    if args.limit:
        tasks = sorted(tasks, key=lambda t: t["id"])[:args.limit]
    if not tasks:
        sys.exit("no task selected")

    t0 = time.monotonic()
    agent = load_agent(args.agent_file)
    load_s = time.monotonic() - t0
    print(f"Agent() loaded in {load_s:.1f} s" + (f"  -- OVER the platform's {LOAD_TIMEOUT_S:.0f} s"
                                                if load_s > LOAD_TIMEOUT_S else ""))
    with tempfile.TemporaryDirectory(prefix="dsh_data_") as root:
        t0 = time.monotonic()
        res = run_agent(LocalProxy(agent), tasks, root, call_timeout=args.timeout,
                        log=print if args.verbose else None)
        took = time.monotonic() - t0
    calls = (len(tasks) + 7) // 8
    print(f"\n{len(tasks)} tasks in {calls} calls, {took:.0f} s ({took / calls:.1f} s per call, "
          f"limit {args.timeout:.0f} s)")
    print("\nscore by type")
    for ty, s in res["type_scores"].items():
        n = sum(t["type"] == ty for t in tasks)
        print(f"  {ty:20s} {s:6.1f}   ({n} tasks)")
    print(f"\n{res['info_message']}")
    print(f"SCORE {res['score']:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"details written to {args.out}")
    if res["run_error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

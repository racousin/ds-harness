"""Evaluate an agent file locally, with the platform's delivery protocol, plus timings and traces.

    python local_eval.py agent_kit_baseline.py                    # dev.json next to this file
    python local_eval.py agent.py --limit 12 --out run.json
    python local_eval.py agent.py --budget-s 330                  # plus a total budget, scaled

This is a thin wrapper around `run_agent` in the challenge's scoring.py (the one next to
the dev file): batching, per-call timeouts, `time_budget_s`, scoring and "one failed call
ends the run" are the platform's own code. On top of it, this script
  * times `Agent()` against the platform's 60 s limit,
  * times every `solve` call against its timeout and keeps the FULL traces (--out),
  * prints the traceback when a call fails (on the platform that run gets no score).
`--limit N` takes N tasks spread over the three levels (N >= 3: the score needs all three).
`--budget-s` is the total budget of the 119-task private mix (plan for 330 s). Like
localtest.py, it is scaled to the tasks you run by the per-task estimates (2 s per
level-1/2 task, 8 s per level-3 task), and by --slowdown; --no-scale uses it as given.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INIT_LIMIT_S = 60.0
PRIVATE_MIX = {1: 37, 2: 67, 3: 15}  # tasks per level in the private set


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def spread(tasks, n):
    """The first n tasks taken round-robin over levels, so every level is present."""
    by_level = {}
    for t in tasks:
        by_level.setdefault(t["level"], []).append(t)
    out, k = [], 0
    while len(out) < n and any(k < len(v) for v in by_level.values()):
        out += [v[k] for _, v in sorted(by_level.items()) if k < len(v)][: n - len(out)]
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="path to an agent file defining class Agent")
    ap.add_argument("--dev", default=os.path.join(HERE, "dev.json"),
                    help="task file; its folder must also hold scoring.py (default: next to this file)")
    ap.add_argument("--limit", type=int, default=0, help="N tasks spread over the levels")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="total time budget for the 119-task private mix, scaled to the tasks run "
                         "(default: none)")
    ap.add_argument("--no-scale", action="store_true", help="use --budget-s as given, unscaled")
    ap.add_argument("--slowdown", type=float, default=1.0,
                    help="multiply the platform's per-task time estimates (and so every call "
                         "timeout) by this factor, e.g. 4 on a laptop or a Colab T4. The "
                         "platform itself uses 1.0: re-check at 1.0 on a GPU like it before submitting")
    ap.add_argument("--out", default=None, help="write the result, timings and full traces here")
    args = ap.parse_args()

    if args.limit and args.limit < 3:
        sys.exit("--limit must be at least 3: the score needs a task of each level")
    scoring = load_module("scoring", os.path.join(os.path.dirname(os.path.abspath(args.dev)), "scoring.py"))
    with open(args.dev) as f:
        tasks = json.load(f)
    for t in tasks:
        scoring.validate_task(t)
    tasks = spread(tasks, args.limit) if args.limit else tasks
    budget = math.inf
    if args.budget_s:
        budget = args.budget_s
        if not args.no_scale:
            def est(counts):
                return sum(scoring.EST_TASK_S[lv] * n for lv, n in counts.items())
            mine = {lv: sum(t["level"] == lv for t in tasks) for lv in PRIVATE_MIX}
            budget *= args.slowdown * est(mine) / est(PRIVATE_MIX)
        print(f"{len(tasks)} tasks: total budget {budget:.0f} s "
              f"({args.budget_s:.0f} s {'as given' if args.no_scale else 'for the private mix, scaled'})")
    if args.slowdown != 1.0:
        for level in scoring.EST_TASK_S:
            scoring.EST_TASK_S[level] *= args.slowdown
        print(f"timeouts scaled by {args.slowdown} (local hardware); the platform uses 1.0")

    class TimedProxy(scoring.LocalProxy):
        """The platform proxy, recording each call's duration, timeout and full replies."""
        calls = []

        def call(self, method, *a, timeout=None, catch_errors=False):
            t0 = time.monotonic()
            out = super().call(method, *a, timeout=timeout, catch_errors=catch_errors)
            self.calls.append({"ids": [t["id"] for t in a[0]], "seconds": time.monotonic() - t0,
                               "timeout": timeout, "replies": out, "error": self.last_error})
            return out

    # The platform puts the agent's directory on sys.path, so `import dsh` works there too.
    agent_path = os.path.abspath(args.agent)
    sys.path.insert(0, os.path.dirname(agent_path))
    t0 = time.monotonic()
    agent = load_module("Agent", agent_path).Agent()
    init_s = time.monotonic() - t0
    print(f"Agent() took {init_s:.1f} s" + ("  WARNING: over the platform's 60 s limit"
                                            if init_s > INIT_LIMIT_S else ""))

    proxy = TimedProxy(agent)
    deadline = time.monotonic() + budget
    res = scoring.run_agent(proxy, tasks, deadline)

    call_of = {tid: c for c in proxy.calls for tid in c["ids"]}
    level = {t["id"]: t["level"] for t in tasks}
    print(f"\n{'id':8s} {'L':1s} {'family':28s} {'score':>5s} {'call s':>7s} {'timeout':>7s}  error")
    for d in res["details"]:
        c = call_of[d["id"]]
        print(f"{d['id']:8s} {level[d['id']]} {d['family']:28s} {d['score']:5.2f} "
              f"{c['seconds']:7.1f} {c['timeout']:7.1f}  {d['error'] or ''}")
    if res["run_error"]:
        print(f"\nRUN ENDED EARLY. On the platform a failed call fails the deployment (no score);\n"
              f"tasks not sent for lack of time score 0:\n"
              f"{res['run_error']}")
    fam_level = {t["family"]: t["level"] for t in tasks}
    print(f"\n{'family':30s} {'level':>5s} {'score':>7s}")
    for fam, s in sorted(res["family_scores"].items(), key=lambda kv: (fam_level[kv[0]], kv[0])):
        print(f"{fam:30s} {fam_level[fam]:5d} {s:7.1f}")
    print(res["info_message"])
    if args.out:
        replies = {r["id"]: r for c in proxy.calls if isinstance(c["replies"], list)
                   for r in c["replies"] if isinstance(r, dict) and "id" in r}
        for d in res["details"]:
            d["trace"] = replies.get(d["id"], {}).get("trace")
            d["answer"] = replies.get(d["id"], {}).get("answer")
            d["call_seconds"] = call_of[d["id"]]["seconds"]
            d["call_index"] = proxy.calls.index(call_of[d["id"]])
        with open(args.out, "w") as f:
            json.dump(dict(res, init_s=init_s, agent=args.agent), f, indent=1, default=str)


if __name__ == "__main__":
    main()

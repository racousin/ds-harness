"""Scoring and delivery of the DS-Harness challenge.

One module for the platform env (env.py) and the local runner (localtest.py):
both call `run_agent`, so a local score and a leaderboard score come from the
same code. Pure stdlib.

A task, as stored in dev.json / private.json
--------------------------------------------
    {"id", "objective", "files": {name: CSV text}, "answer", "tol", "type"}

What the agent receives, in batches of 8, through `Agent.solve(tasks)`:
    {"id": "d_042", "objective": "...", "files": ["/.../data/d_042/loans.csv", ...]}
`files` lists absolute paths, empty when the task has none. The files of a
batch are written before the call and deleted after it.

What the agent returns: a list of {"id": ..., "answer": <one number>}.
An answer is an int or a float (not a bool, not NaN), or a string holding a
plain decimal number ("41.07", "-3", "1e-3"). Anything else is a format error.

Score
-----
A task scores 1 when |answer - gold| <= tol, else 0; an unanswered task scores 0.
The objective states the rounding: "n decimals" gives tol = 1.5 * 10**-n,
"an integer" gives tol = 1e-6. Score = 100 x the mean over all tasks.

Timing
------
The leaderboard run sends the 120 private tasks as 15 calls of 8. Each call must
return within CALL_TIMEOUT_S. A call that raises or runs out of time ends the
run: the tasks not answered score 0, and on the platform the deployment fails.
"""
import concurrent.futures
import json
import math
import os
import random
import re
import shutil
import traceback

BATCH_SIZE = 8
CALL_TIMEOUT_S = 40.0
LOAD_TIMEOUT_S = 60.0          # Agent() must load its model within this (enforced by the platform)
TEST_RUN_CALLS = 2             # the platform's test run: 2 calls of dev tasks
ORDER_SEED = 4242              # one delivery order, the same for every agent
ECHO_CHARS = 160               # agent text quoted in info_message, at most

AGENT_KEYS = ("id", "objective", "files")
GOLD_KEYS = ("answer", "tol", "type")
FAMILIES = ("calc", "table", "fit")
METRIC_KEYS = ("calc", "table", "fit", "tasks_answered", "format_errors")

_PLAIN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class AnswerFormatError(ValueError):
    """The answer is not one number."""


def family(task):
    return task["type"].split(".")[0]


def validate_task(task):
    """Fail fast on a malformed task (called when a split is loaded)."""
    for k in AGENT_KEYS + GOLD_KEYS:
        if k not in task:
            raise ValueError(f"task {task.get('id')!r} lacks key {k!r}")
    if not isinstance(task["files"], dict):
        raise ValueError(f"task {task['id']}: files must map a file name to its text")
    for name in task["files"]:
        if os.path.basename(name) != name or name.startswith("."):
            raise ValueError(f"task {task['id']}: bad file name {name!r}")
    if family(task) not in FAMILIES:
        raise ValueError(f"task {task['id']}: unknown type {task['type']!r}")
    float(task["answer"]), float(task["tol"])


def coerce_answer(x):
    if isinstance(x, bool):
        raise AnswerFormatError("a boolean is not a number")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        s = x.strip()
        if not _PLAIN.match(s):
            raise AnswerFormatError(f"not a plain number: {x[:40]!r}")
        v = float(s)
    else:
        raise AnswerFormatError(f"expected a number, got {type(x).__name__}")
    if not math.isfinite(v):
        raise AnswerFormatError(f"non-finite number {x!r}")
    return v


def score_answer(task, answer):
    """(1.0 or 0.0, format error message or None)."""
    try:
        v = coerce_answer(answer)
    except AnswerFormatError as e:
        return 0.0, str(e)
    return (1.0 if abs(v - float(task["answer"])) <= float(task["tol"]) else 0.0), None


# --- delivery -------------------------------------------------------------------

def make_batches(tasks, batch_size=BATCH_SIZE):
    """A fixed shuffle (types mixed), cut into batches."""
    order = sorted(tasks, key=lambda t: t["id"])
    random.Random(ORDER_SEED).shuffle(order)
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def test_run_subset(tasks, n=TEST_RUN_CALLS * BATCH_SIZE):
    """The dev tasks of the platform's test run: one task per type in turn, `n` in all."""
    by_type = {}
    for t in sorted(tasks, key=lambda t: t["id"]):
        by_type.setdefault(t["type"], []).append(t)
    out, i = [], 0
    while len(out) < n and any(len(v) > i for v in by_type.values()):
        for ty in sorted(by_type):
            if i < len(by_type[ty]) and len(out) < n:
                out.append(by_type[ty][i])
        i += 1
    return out


def write_files(batch, data_root):
    """Write each task's files under data_root/<id>/ and return the agent payload."""
    payload = []
    for t in batch:
        paths = []
        if t["files"]:
            d = os.path.join(data_root, t["id"])
            os.makedirs(d, mode=0o755, exist_ok=True)
            for name, text in t["files"].items():
                p = os.path.join(d, name)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(text)
                os.chmod(p, 0o644)
                paths.append(p)
        payload.append({"id": t["id"], "objective": t["objective"], "files": paths})
    return payload


def delete_files(batch, data_root):
    for t in batch:
        shutil.rmtree(os.path.join(data_root, t["id"]), ignore_errors=True)


class LocalProxy:
    """Local stand-in for the platform AgentProxy: `.call(method, *args, timeout=,
    catch_errors=True)`, `.last_error`. As on the platform, the payload and the
    reply go through JSON, and after one failure (exception or missed timeout)
    the agent is broken: later calls fail without running."""

    def __init__(self, agent):
        self._agent = agent
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.last_error = None
        self._broken = None

    def call(self, method, *args, timeout=None, catch_errors=False):
        self.last_error = None
        if self._broken is None:
            args = json.loads(json.dumps(list(args)))
            future = self._pool.submit(getattr(self._agent, method), *args)
            try:
                return json.loads(json.dumps(future.result(timeout=timeout), default=_json_default))
            except concurrent.futures.TimeoutError:
                self._broken = f"call {method!r} deadline exceeded ({timeout:.0f} s)"
            except Exception as e:  # the agent's own exception ends the run, as on the platform
                self._broken = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        self.last_error = self._broken
        if catch_errors:
            return None
        raise RuntimeError(self._broken)


def _json_default(obj):
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"solve() returned a value that is not JSON: {type(obj).__name__}")


def _read_replies(replies, batch):
    """id -> answer for one batch; (answers, error count, first error)."""
    if not isinstance(replies, list):
        return {}, len(batch), f"solve() returned {type(replies).__name__}, expected a list"
    ids = {t["id"] for t in batch}
    answers, errors, first = {}, 0, None
    for r in replies:
        if not isinstance(r, dict) or "id" not in r or "answer" not in r:
            errors += 1
            first = first or "a reply element is not a dict with keys 'id' and 'answer'"
        elif r["id"] not in ids:
            errors += 1
            first = first or f"unknown id {str(r['id'])[:40]!r} in the reply"
        elif r["id"] in answers:
            errors += 1
            first = first or f"id {r['id']!r} answered twice (the first answer counts)"
        else:
            answers[r["id"]] = r["answer"]
    return answers, errors, first


def run_agent(proxy, tasks, data_root, call_timeout=CALL_TIMEOUT_S, log=None):
    """Deliver `tasks` to one agent, batch after batch, until done or a failed call."""
    per_task, details = {}, []
    format_errors, answered = 0, 0
    first_error, run_error = None, None
    for batch in make_batches(tasks):
        payload = write_files(batch, data_root)
        try:
            replies = proxy.call("solve", payload, timeout=call_timeout, catch_errors=True)
        finally:
            delete_files(batch, data_root)
        if proxy.last_error is not None:
            run_error = proxy.last_error
            break
        answers, errs, first = _read_replies(replies, batch)
        format_errors += errs
        first_error = first_error or first
        for t in batch:
            if t["id"] in answers:
                answered += 1
                s, err = score_answer(t, answers[t["id"]])
            else:
                s, err = 0.0, "no answer"
            if err and err != "no answer":
                format_errors += 1
                first_error = first_error or f"{t['id']}: {err}"
            per_task[t["id"]] = s
            details.append({"id": t["id"], "type": t["type"], "score": s, "error": err,
                            "answer": answers.get(t["id"]), "gold": t["answer"]})
            if log is not None:
                log(f"{t['id']} {t['type']:18s} {s:.0f} gold {t['answer']!s:>12} "
                    f"got {str(answers.get(t['id']))[:24]!s:>12} {err or ''}")
    return summarize(tasks, per_task, details, answered, format_errors, first_error, run_error)


def summarize(tasks, per_task, details, answered, format_errors, first_error, run_error):
    n = len(tasks)
    score = 100.0 * sum(per_task.values()) / n
    fam, typ = {}, {}
    for t in tasks:
        s = per_task.get(t["id"], 0.0)
        fam.setdefault(family(t), []).append(s)
        typ.setdefault(t["type"], []).append(s)
    fam_scores = {f: 100.0 * sum(v) / len(v) for f, v in fam.items()}
    type_scores = {ty: 100.0 * sum(v) / len(v) for ty, v in sorted(typ.items())}
    metrics = {f: round(fam_scores.get(f, 0.0), 3) for f in FAMILIES}
    metrics.update(tasks_answered=answered, format_errors=format_errors)
    info = (f"score {score:.1f} | no files {fam_scores.get('calc', 0):.1f}, tables "
            f"{fam_scores.get('table', 0):.1f}, fits {fam_scores.get('fit', 0):.1f} | "
            f"{answered}/{n} answered, {format_errors} format errors")
    if first_error:
        info += f" | first format error: {first_error[:ECHO_CHARS]}"
    if run_error:
        last = [ln for ln in run_error.strip().splitlines() if ln.strip()]
        info += f" | run ended: {(last[-1] if last else run_error)[:ECHO_CHARS]}"
    return {"score": round(score, 4), "metrics_detail": metrics, "info_message": info,
            "type_scores": type_scores, "run_error": run_error, "details": details}

"""Scoring and evaluation protocol of the DS-Harness challenge (v1).

One module, shipped to students as-is: the platform env (env.py) and the
local runner (localtest.py) both call `run_agent`, so local and platform
scores come from one code path. Pure stdlib + numpy.

Answer coercion (anything else raises AnswerFormatError)
--------------------------------------------------------
number       int / float / numpy integer or floating scalar (not bool, not
             NaN/inf), or a string: optional sign (ASCII or Unicode minus),
             optional currency ($, EUR sign, GBP sign, "EUR"), optional
             trailing "%" (removed, value kept: "12.5%" -> 12.5), then a
             plain decimal ("12.5", "1e-3"), a decimal with comma thousands
             separators ("1,200.50") or a fraction of integers ("5/36").
             "12,5" is ambiguous and rejected.
category     a string (or an integer, read as its decimal string); compared
             after strip() + casefold().
list         a list, tuple or 1-D numpy array of category labels, compared
             element by element in order (casefolded).
vector /     a list, tuple or 1-D numpy array of exactly `scoring["length"]`
predictions  elements: numbers (smape, mae, brier) or labels (macro_f1,
             casefolded).

Per-task score in [0, 1]
------------------------
numeric      1 iff |answer - gold| <= abs_tol + rel_tol * |gold|.
category     1 iff labels match. list: 1 iff every position matches.
normalized   (m - baseline) / (reference - baseline) clipped to [0, 1], for
             m in {smape, mae, ape, brier, macro_f1}. Baseline = a trivial
             rule, reference = a tuned solution, both stored in the task.

Leaderboard score
-----------------
100 * (0.3 L1 + 0.4 L2 + 0.3 L3), each level = mean task score over the
level's tasks. Unanswered tasks score 0.
"""
import concurrent.futures
import json
import math
import random
import re
import time
import traceback

import numpy as np

# --- protocol constants -------------------------------------------------------

LEVEL_WEIGHTS = {1: 0.3, 2: 0.4, 3: 0.3}
METRIC_KEYS = ("level_1", "level_2", "level_3", "heldout", "format_errors", "tasks_answered")
AGENT_KEYS = ("id", "prompt", "files", "answer_type")
GOLD_KEYS = ("family", "level", "answer", "scoring", "heldout_family")
ANSWER_TYPES = ("number", "category", "list", "vector", "predictions")
BATCH_SIZE = 8              # L1/L2 tasks per call
L3_BATCH_SIZE = 2           # L3 tasks per call (own batches)
EST_TASK_S = {1: 2.0, 2: 2.0, 3: 8.0}   # expected seconds per task for a 3B harness
BUDGET_FACTOR = 2.5         # batch budget = factor * sum of the estimates of its tasks
MARGIN_S = 5.0              # kept free before the deadline
MIN_CALL_S = 2.0            # a call shorter than this is not attempted
ORDER_SEED = 4242           # same delivery order for every agent
ECHO_CHARS = 160            # agent text quoted in info_message, at most

_PLAIN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")
_CURRENCY = ("$", "€", "£")


class AnswerFormatError(ValueError):
    """The answer cannot be read under the coercion rules above."""


# --- coercion -----------------------------------------------------------------

def coerce_number(x):
    if isinstance(x, (bool, np.bool_)):
        raise AnswerFormatError("a boolean is not a number")
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = float(x)
        except OverflowError:
            raise AnswerFormatError(f"number too large: {str(x)[:40]}...") from None
    elif isinstance(x, str):
        try:
            v = _parse_number_string(x)
        except (OverflowError, ValueError) as e:  # huge fractions, >4300-digit integers
            if isinstance(e, AnswerFormatError):
                raise
            raise AnswerFormatError(f"cannot read a number from {x[:40]!r}...") from None
    else:
        raise AnswerFormatError(f"expected a number, got {type(x).__name__}")
    if not math.isfinite(v):
        raise AnswerFormatError(f"non-finite number {x!r}")
    return v


def _parse_number_string(raw):
    s = raw.strip().replace("−", "-")
    sign = ""
    if s[:1] in "+-" and len(s) > 1:
        sign, s = s[0], s[1:].strip()
    if s.upper().endswith("EUR"):
        s = s[:-3].strip()
    if s[:1] in _CURRENCY:
        s = s[1:].strip()
    elif s[-1:] in _CURRENCY:
        s = s[:-1].strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    s = sign + s
    if _PLAIN.match(s):
        return float(s)
    if _THOUSANDS.match(s):
        return float(s.replace(",", ""))
    m = _FRACTION.match(s)
    if m:
        den = int(m.group(2))
        if den == 0:
            raise AnswerFormatError(f"zero denominator in {raw!r}")
        return int(m.group(1)) / den
    raise AnswerFormatError(f"cannot read a number from {raw!r}")


def coerce_label(x):
    if isinstance(x, str):
        return x.strip().casefold()
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        try:
            return str(int(x))
        except ValueError:  # > 4300 digits
            raise AnswerFormatError("integer label too long") from None
    raise AnswerFormatError(f"expected a string label, got {type(x).__name__}")


def coerce_list(x, length=None):
    if isinstance(x, np.ndarray):
        if x.ndim != 1:
            raise AnswerFormatError(f"expected a 1-D array, got shape {x.shape}")
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        raise AnswerFormatError(f"expected a list, got {type(x).__name__}")
    if length is not None and len(x) != length:
        raise AnswerFormatError(f"expected {length} values, got {len(x)}")
    return list(x)


# --- metrics ------------------------------------------------------------------

def smape(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    denom = np.abs(pred) + np.abs(actual)
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(pred - actual) / np.where(denom == 0, 1, denom))
    return float(np.mean(ratio) * 100)


def mae(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def ape(pred, actual):
    """Absolute percentage error of one number (a forecast total)."""
    return float(100 * abs(float(pred) - float(actual)) / abs(float(actual)))


def brier(prob, actual01):
    p, y = np.asarray(prob, float), np.asarray(actual01, float)
    return float(np.mean((p - y) ** 2))


def macro_f1(pred, actual, labels):
    """Macro-F1 over `labels`; all labels are compared casefolded."""
    pred = np.asarray([str(p).strip().casefold() for p in pred], dtype=object)
    actual = np.asarray([str(a).strip().casefold() for a in actual], dtype=object)
    f1s = []
    for c in (str(lab).strip().casefold() for lab in labels):
        tp = np.sum((pred == c) & (actual == c))
        fp = np.sum((pred == c) & (actual != c))
        fn = np.sum((pred != c) & (actual == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def normalize(m, baseline, reference):
    """(m - baseline) / (reference - baseline), clipped to [0, 1]."""
    if reference == baseline:
        raise ValueError("reference equals baseline: normalisation undefined")
    return float(min(1.0, max(0.0, (m - baseline) / (reference - baseline))))


NORMALIZED_METRICS = ("smape", "mae", "ape", "brier", "macro_f1")


# --- task validation and scoring ------------------------------------------------

def validate_task(task):
    """Fail fast on a malformed task (called when a split is loaded)."""
    for k in AGENT_KEYS + GOLD_KEYS:
        if k not in task:
            raise ValueError(f"task {task.get('id')!r} lacks key {k!r}")
    if task["answer_type"] not in ANSWER_TYPES:
        raise ValueError(f"task {task['id']}: bad answer_type {task['answer_type']!r}")
    if task["level"] not in LEVEL_WEIGHTS:
        raise ValueError(f"task {task['id']}: bad level {task['level']!r}")
    sc = task["scoring"]
    kind = sc["kind"]
    if kind == "numeric":
        float(sc["abs_tol"]), float(sc["rel_tol"]), float(task["answer"])
    elif kind in ("category", "list"):
        pass
    elif kind == "normalized":
        if sc["metric"] not in NORMALIZED_METRICS:
            raise ValueError(f"task {task['id']}: unknown metric {sc['metric']!r}")
        if sc["reference"] == sc["baseline"]:
            raise ValueError(f"task {task['id']}: reference equals baseline")
        if sc["metric"] != "ape" and len(task["answer"]) != sc["length"]:
            raise ValueError(f"task {task['id']}: gold length differs from scoring length")
        if sc["metric"] == "macro_f1":
            sc["labels"]
        if sc["metric"] == "brier":
            sc["positive"]
    else:
        raise ValueError(f"task {task['id']}: unknown scoring kind {kind!r}")


def score_answer(task, answer):
    """Return (score in [0, 1], format error message or None)."""
    sc = task["scoring"]
    kind = sc["kind"]
    try:
        if kind == "numeric":
            v = coerce_number(answer)
            gold = float(task["answer"])
            return (1.0 if abs(v - gold) <= sc["abs_tol"] + sc["rel_tol"] * abs(gold) else 0.0), None
        if kind == "category":
            return (1.0 if coerce_label(answer) == coerce_label(task["answer"]) else 0.0), None
        if kind == "list":
            got = [coerce_label(a) for a in coerce_list(answer)]
            gold = [coerce_label(a) for a in task["answer"]]
            return (1.0 if got == gold else 0.0), None
        if kind == "normalized":
            metric = sc["metric"]
            if metric == "ape":
                m = ape(coerce_number(answer), task["answer"])
            elif metric == "macro_f1":
                pred = [coerce_label(a) for a in coerce_list(answer, sc["length"])]
                m = macro_f1(pred, task["answer"], sc["labels"])
            elif metric == "brier":
                p = [coerce_number(a) for a in coerce_list(answer, sc["length"])]
                if min(p) < 0 or max(p) > 1:
                    raise AnswerFormatError("probabilities must lie in [0, 1]")
                y = [1.0 if coerce_label(a) == coerce_label(sc["positive"]) else 0.0
                     for a in task["answer"]]
                m = brier(p, y)
            else:
                pred = [coerce_number(a) for a in coerce_list(answer, sc["length"])]
                m = {"smape": smape, "mae": mae}[metric](pred, task["answer"])
            return normalize(m, sc["baseline"], sc["reference"]), None
    except AnswerFormatError as e:
        return 0.0, str(e)
    raise ValueError(f"unknown scoring kind {kind!r}")


# --- aggregation ----------------------------------------------------------------

def aggregate(tasks, per_task):
    """Leaderboard score (0-100), level means over tasks (0-100), family means
    (0-100, reported only), held-out-family mean (0-100, 0 when the split has none)."""
    by_family, by_level, held = {}, {}, []
    for t in tasks:
        s = per_task.get(t["id"], 0.0)
        by_family.setdefault(t["family"], []).append(s)
        by_level.setdefault(t["level"], []).append(s)
        if t["heldout_family"]:
            held.append(s)
    family_scores = {f: 100 * sum(v) / len(v) for f, v in sorted(by_family.items())}
    level_scores = {}
    for lv in LEVEL_WEIGHTS:
        if lv not in by_level:
            raise ValueError(f"level {lv} has no task: the split is malformed")
        level_scores[lv] = 100 * sum(by_level[lv]) / len(by_level[lv])
    score = sum(LEVEL_WEIGHTS[lv] * level_scores[lv] for lv in LEVEL_WEIGHTS)
    heldout = 100 * sum(held) / len(held) if held else 0.0
    return score, level_scores, family_scores, heldout


# --- delivery -------------------------------------------------------------------

def make_batches(tasks):
    """L1/L2 tasks in batches of BATCH_SIZE, L3 tasks in batches of
    L3_BATCH_SIZE, L3 batches spread evenly among the others."""
    rng = random.Random(ORDER_SEED)
    small = [t for t in tasks if t["level"] != 3]
    big = [t for t in tasks if t["level"] == 3]
    rng.shuffle(small)
    rng.shuffle(big)
    sb = [small[i:i + BATCH_SIZE] for i in range(0, len(small), BATCH_SIZE)]
    bb = [big[i:i + L3_BATCH_SIZE] for i in range(0, len(big), L3_BATCH_SIZE)]
    keyed = [((i + 1) / (len(sb) + 1), 0, b) for i, b in enumerate(sb)]
    keyed += [((j + 1) / (len(bb) + 1), 1, b) for j, b in enumerate(bb)]
    return [b for _, _, b in sorted(keyed, key=lambda x: (x[0], x[1]))]


def batch_budget(batch):
    return BUDGET_FACTOR * sum(EST_TASK_S[t["level"]] for t in batch)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"agent 'solve' returned non-JSON-serializable value: "
                    f"Object of type {type(obj).__name__} is not JSON serializable")


class LocalProxy:
    """Local stand-in for the platform AgentProxy: `.call(method, *args,
    timeout=, catch_errors=True)`, `.last_error`. Like the platform, the
    payload goes through JSON, and after one failure (exception or missed
    deadline) the agent is broken: later calls fail without running."""

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
                result = future.result(timeout=timeout)
                # the platform ships the reply back as JSON (numpy arrays/scalars allowed):
                # anything else (set, Decimal, pandas objects, ...) ends the run there too
                return json.loads(json.dumps(result, default=_json_default))
            except concurrent.futures.TimeoutError:
                self._broken = f"call {method!r} deadline exceeded ({timeout:.1f} s)"
            except Exception as e:  # the agent's own exception is reported, not handled
                self._broken = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        self.last_error = self._broken
        if catch_errors:
            return None
        raise RuntimeError(self._broken)


def _read_replies(replies, batch):
    """Map id -> answer for one batch. Returns (answers, duplicated ids,
    error count, first error message)."""
    if isinstance(replies, np.ndarray):
        replies = replies.tolist()
    if not isinstance(replies, list):
        return {}, set(), 0, f"solve() returned {type(replies).__name__}, expected a list"
    ids = {t["id"] for t in batch}
    answers, dup, errors, first = {}, set(), 0, None
    for r in replies:
        if not isinstance(r, dict) or "id" not in r or "answer" not in r:
            errors += 1
            first = first or "a reply element is not a dict with keys 'id' and 'answer'"
            continue
        rid = r["id"]
        if not isinstance(rid, str) or rid not in ids:
            errors += 1
            first = first or f"unknown id {rid!r} in the reply"
            continue
        if rid in answers:
            dup.add(rid)
            continue
        answers[rid] = r["answer"]
    return answers, dup, errors, first


def run_agent(proxy, tasks, deadline_monotonic, clock=time.monotonic, log=None):
    """Deliver `tasks` to one agent until done, a failed call, or the deadline.

    Each call gets timeout = its batch budget; a batch whose budget no longer
    fits in the time left - MARGIN_S is not sent (its tasks score 0). Each task
    dict carries `time_budget_s`, its share of that timeout. A failed call ends
    the agent's run: tasks not answered score 0. On the platform a failed call
    also fails the deployment."""
    per_task, details = {}, []
    format_errors, answered, sent = 0, 0, 0
    first_format_error, run_error = None, None
    for batch in make_batches(tasks):
        left = deadline_monotonic - clock() - MARGIN_S
        if left < MIN_CALL_S:
            run_error = run_error or "time budget exhausted; remaining tasks not sent"
            break
        # Every call gets its full budget: a batch whose budget no longer fits
        # is not sent, so a call's timeout is never shortened near the deadline.
        timeout = batch_budget(batch)
        if timeout > left:
            run_error = run_error or "time budget exhausted; remaining tasks not sent"
            continue
        est = sum(EST_TASK_S[t["level"]] for t in batch)
        payload = [dict({k: t[k] for k in AGENT_KEYS},
                        time_budget_s=round(timeout * EST_TASK_S[t["level"]] / est, 1))
                   for t in batch]
        replies = proxy.call("solve", payload, timeout=timeout, catch_errors=True)
        if proxy.last_error is not None:
            run_error = proxy.last_error
            break
        sent += len(batch)
        answers, dup, errs, first = _read_replies(replies, batch)
        format_errors += errs
        first_format_error = first_format_error or first
        traces = {r["id"]: r.get("trace") for r in (replies if isinstance(replies, list) else [])
                  if isinstance(r, dict) and isinstance(r.get("id"), str)}
        for t in batch:
            tid = t["id"]
            if tid in dup:
                s, err = 0.0, "duplicate id in the reply"
            elif tid not in answers:
                s, err = 0.0, "no answer returned"
            else:
                answered += 1
                s, err = score_answer(t, answers[tid])
            if err is not None:
                format_errors += 1
                first_format_error = first_format_error or f"{tid}: {err}"
            per_task[tid] = s
            details.append({"id": tid, "family": t["family"], "score": round(s, 4),
                            "error": err, "trace": str(traces.get(tid))[:300]})
            if log is not None:
                log(f"{tid} {t['family']:28s} {s:.2f} {err or ''} | {str(traces.get(tid))[:160]}")
    score, level_scores, family_scores, heldout = aggregate(tasks, per_task)
    metrics = {"level_1": round(level_scores[1], 3), "level_2": round(level_scores[2], 3),
               "level_3": round(level_scores[3], 3), "heldout": round(heldout, 3),
               "format_errors": format_errors, "tasks_answered": answered}
    info = (f"score {score:.1f} | L1 {level_scores[1]:.1f} L2 {level_scores[2]:.1f} "
            f"L3 {level_scores[3]:.1f} held-out {heldout:.1f} | {answered}/{len(tasks)} answered, "
            f"{sent} sent, {format_errors} format errors")
    # Agent-controlled text is capped: info_message is shown with the run.
    if first_format_error:
        info += f" | first format error: {first_format_error[:ECHO_CHARS]}"
    if run_error:
        last = [ln for ln in run_error.strip().splitlines() if ln.strip()]
        info += f" | run ended: {(last[-1] if last else run_error)[:ECHO_CHARS]}"
    return {"score": round(score, 4), "metrics_detail": metrics, "info_message": info,
            "level_scores": level_scores, "family_scores": family_scores,
            "run_error": run_error, "details": details}

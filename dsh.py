"""dsh -- the DS-Harness starter kit (one file, upload it next to your agent.py).

What this module gives you (the plumbing that is fiddly or dangerous to get right):

    load_llm(model_id)          a small chat model, batched generation (revision pinned
                                to the cached snapshot: see cached_revision;
                                trust_remote_code on for REMOTE_CODE_MODELS)
    run_python(code, files)     run model-written code in a subprocess, read back RESULT
    calculator(expr)            a safe arithmetic evaluator (the "hello world" tool)
    csv_overview(files)         a compact text description of the task's CSV files
    coerce_number / coerce_label / coerce_list / to_answer
                                the challenge scorer's coercion rules, mirrored
    parse_number / parse_label / parse_list
                                tolerant extraction of an answer from free LLM text
    Budget                      a stopwatch against the tasks' time_budget_s

What it does NOT give you (that is your project): the router, the prompts and few-shot
banks, which tools the model may call and how, retries, verification, self-consistency,
model choice, fine-tuning. See README.md.

Platform facts this file is written for: no network (models come from a read-only HF
cache), read-only root filesystem, /tmp is a 128 MB tmpfs, 3 CPUs and 3 GiB of RAM for
the whole agent container, one GPU shared with nobody else during your job.
"""
from __future__ import annotations

import ast
import io
import json
import math
import operator
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

# ============================================================================ LLM


class LLM:
    """A causal LM plus its tokenizer, with one method: `chat`."""

    def __init__(self, model, tokenizer, device: str):
        self.model, self.tok, self.device = model, tokenizer, device
        # Qwen3-style templates accept `enable_thinking`. We switch thinking OFF by default:
        # a 1-3B model thinking for 2000 tokens per task does not fit the time budget.
        self.has_thinking_flag = "enable_thinking" in (tokenizer.chat_template or "")

    def chat(self, batches: list[list[dict]], max_new_tokens: int = 512,
             temperature: float = 0.0, batch_size: int = 8, thinking: bool = False,
             max_time: float | None = None) -> list[str]:
        """Generate one reply per conversation.

        batches: a list of conversations, each a list of {"role", "content"} messages.
        Returns the decoded replies, in the same order as the input.
        Greedy decoding when temperature == 0 (deterministic), sampling otherwise.
        max_time: wall-clock seconds for the whole call. Generation stops early when it runs
        out (replies are then truncated, and sub-batches not started come back as "").
        A generate() call cannot be interrupted from outside, so this is how you keep it
        inside a time budget.
        """
        import torch

        kw = {"enable_thinking": thinking} if self.has_thinking_flag else {}
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                 for m in batches]
        # Sort by length so each sub-batch pads as little as possible, then restore the order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        replies: list[str] = [""] * len(texts)
        gen = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen.update(do_sample=False)
        t_end = None if max_time is None else time.monotonic() + max_time
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            if t_end is not None:
                left = t_end - time.monotonic()
                if left <= 0.5:
                    break
                gen["max_time"] = left  # transformers' built-in MaxTimeCriteria
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen)
            new = out[:, enc["input_ids"].shape[1]:]  # left padding: prompts all end at the same column
            for i, text in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                replies[i] = text
        return replies


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cached_revision(model_id: str) -> str | None:
    """The commit of `model_id` held in the local HF cache, or None when it is not cached.

    The platform's upload scan (bandit B615) refuses a `from_pretrained` call without a
    pinned `revision`. Pinning the snapshot the cache holds satisfies it and loads exactly
    the weights the platform has. On Colab, a model not cached yet gives None: it is
    downloaded on first use, and cached from then on.
    """
    from huggingface_hub import constants

    ref = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"), "refs", "main")
    if not os.path.exists(ref):
        return None
    with open(ref) as f:
        return f.read().strip()


# Models of the platform cache that ship their own modelling code: they load only with
# trust_remote_code=True. The code comes from the same pinned snapshot as the weights.
REMOTE_CODE_MODELS = frozenset({
    "IndexTeam/Index-1.9B-Chat",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "internlm/internlm2_5-1_8b-chat",
    "openbmb/MiniCPM-2B-sft-bf16",
})


def load_llm(model_id: str, device: str | None = None,
             trust_remote_code: bool | None = None) -> LLM:
    """Load a chat model from the local HF cache.

    fp16 on cuda / mps (2 bytes per parameter), fp32 on cpu (fp16 matmuls are slow there).
    The weights are loaded straight onto the target device (`device_map`), so they never sit
    in full in CPU RAM: the agent container has 3 GiB of RAM, a 3B model is 6 GB in fp16.
    (transformers >= 5 always loads lazily; older versions needed low_cpu_mem_usage=True.)
    trust_remote_code: None (default) turns it on for the ids in REMOTE_CODE_MODELS only.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    revision = cached_revision(model_id)
    if trust_remote_code is None:
        trust_remote_code = model_id in REMOTE_CODE_MODELS
    extra = {"trust_remote_code": True} if trust_remote_code else {}
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, **extra)
    tok.padding_side = "left"  # decoder-only models generate after the last token: pad on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                     device_map="cuda", **extra)
    else:
        # device_map="mps" crashed (segfault) with torch 2.14 / transformers 5.17 on macOS;
        # a Mac has unified memory anyway, so load on CPU and move.
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                     dtype=dtype, **extra).to(device)
    model.eval()
    return LLM(model, tok, device)


# ============================================================================ sandbox


@dataclass
class ExecResult:
    ok: bool            # True iff the code ran and RESULT is a JSON value (not None)
    value: object       # RESULT, converted to plain Python (numpy/pandas -> float/list)
    stdout: str         # what the code printed (truncated)
    error: str | None   # traceback tail, timeout or "no RESULT" message; feed it back to the model
    seconds: float


# The runner executes solution.py and writes RESULT to result.json. It is strict on
# purpose: only a variable named RESULT counts. Small models often write `result = ...`
# or end with a bare expression `df.mean()`; accepting those is a leniency YOU may add
# (e.g. rewrite the code, or extend this runner in your own copy) and discuss at the oral.
_RUNNER = r"""
import json, sys, traceback
ns = {"__name__": "__main__"}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
except BaseException:
    traceback.print_exc()
    sys.exit(1)
if "RESULT" not in ns:
    near = [k for k in ("result", "answer", "res", "ans") if k in ns]
    hint = f" (you defined {near[0]!r}; rename it RESULT)" if near else ""
    sys.stderr.write("NameError: no variable RESULT was defined" + hint + "\n")
    sys.exit(1)

def plain(x):
    if hasattr(x, "tolist"):          # numpy scalar/array, pandas Series/Index
        return plain(x.tolist())
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    return x

try:
    json.dump(plain(ns["RESULT"]), open("result.json", "w"), allow_nan=True)
except (TypeError, ValueError) as e:
    sys.stderr.write(f"TypeError: RESULT of type {type(ns['RESULT']).__name__} is not a "
                     f"number, string or list ({e})\n")
    sys.exit(1)
"""


def _limit_memory(mb: int):
    """Return a preexec_fn capping the child's address space (Linux only: macOS ignores
    RLIMIT_AS, Windows has no `resource`). A runaway `np.zeros((10**6, 10**6))` then fails
    with MemoryError in the child instead of OOM-killing your whole agent."""
    if not sys.platform.startswith("linux"):
        return None

    def fn():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mb << 20, mb << 20))
    return fn


def run_python(code: str, files: dict[str, str], timeout_s: float = 20.0,
               workdir_root: str = "/tmp", mem_limit_mb: int = 1024,
               max_output: int = 4000) -> ExecResult:
    """Run `code` in a fresh python subprocess whose working directory holds the task's CSVs.

    The code must assign its answer to a variable named RESULT. Cost: a cold python +
    pandas start is ~0.5-1 s on the platform's 3 CPUs, so budget for it.
    The temp dir is created under `workdir_root` (/tmp is the only writable place on the
    platform, 128 MB) and deleted afterwards, whatever happens.
    """
    t0 = time.monotonic()
    d = tempfile.mkdtemp(prefix="dsh_", dir=workdir_root)
    try:
        for name, text in files.items():
            if os.path.basename(name) != name:
                raise ValueError(f"file name {name!r} must not contain a path")
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "_runner.py"), "w") as f:
            f.write(_RUNNER)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=d, PYTHONDONTWRITEBYTECODE="1")
        # start_new_session: the child leads its own process group, so on timeout we kill it
        # and anything it spawned (joblib workers...) in one go.
        p = subprocess.Popen([sys.executable, "_runner.py"], cwd=d, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, preexec_fn=_limit_memory(mem_limit_mb))
        try:
            out, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return ExecResult(False, None, out[-max_output:],
                              f"TimeoutError: the code ran longer than {timeout_s:.0f} s",
                              time.monotonic() - t0)
        out = out[-max_output:]
        if p.returncode != 0:
            return ExecResult(False, None, out, err[-max_output:] or f"exit code {p.returncode}",
                              time.monotonic() - t0)
        with open(os.path.join(d, "result.json")) as f:
            value = json.load(f)
        if value is None:
            return ExecResult(False, None, out, "RESULT is None: assign the final answer to RESULT",
                              time.monotonic() - t0)
        return ExecResult(True, value, out, None, time.monotonic() - t0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ============================================================================ calculator

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "exp", "log", "log10", "log2", "sin", "cos", "tan", "floor", "ceil",
    "factorial", "comb", "perm", "fabs")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def calculator(expr: str) -> float:
    """Evaluate an arithmetic expression safely: numbers, + - * / // % **, parentheses,
    pi, e and a whitelist of math functions. No names, attributes or imports, so a model
    cannot run arbitrary code through it. Raises ValueError on anything else."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 1000:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](a, b)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            raise ValueError("'^' is not a power in Python: use **")
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            args = [ev(a) for a in node.args]
            if node.func.id == "factorial" and args and args[0] > 1000:
                raise ValueError("factorial argument too large")
            return _FUNCS[node.func.id](*args)
        raise ValueError(f"not allowed in a calculator expression: {ast.dump(node)[:60]}")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse {expr!r}: {e.msg}") from e
    return float(ev(tree))


# ============================================================================ CSV overview


def csv_overview(files: dict[str, str], n_rows: int = 3, show_all_below: int = 25) -> str:
    """One paragraph per file: shape, columns with dtypes (and missing counts), first rows.
    This is what the model needs to write correct column names; the full CSV would not
    fit a small model's context. Files with fewer than `show_all_below` rows (a
    data_dictionary.csv, a small lookup table) are shown in full: they are the documentation."""
    import pandas as pd
    parts = []
    for name, text in files.items():
        df = pd.read_csv(io.StringIO(text))
        cols = []
        for c, t in df.dtypes.astype(str).items():
            na = int(df[c].isna().sum())
            cols.append(f"{c} ({t}{f', {na} missing' if na else ''})")
        parts.append(f"File `{name}`: {df.shape[0]} rows x {df.shape[1]} columns. "
                     f"Columns: {', '.join(cols)}.\n"
                     + (f"First rows:\n{df.head(n_rows).to_csv(index=False).strip()}"
                        if len(df) >= show_all_below else f"All rows:\n{text.strip()}"))
    return "\n\n".join(parts)


# ============================================================================ answers
# Mirror of the challenge scorer's coercion rules (public/scoring.py, v1: numpy scalars and
# arrays accepted, Unicode minus, labels compared casefolded). test_dsh.py checks the two
# agree; if they ever disagree, the scorer wins.

_PLAIN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")
_CURRENCY = ("$", "€", "£")


def coerce_number(x) -> float:
    """Exactly what the scorer accepts for a number: int/float (not bool, not NaN/inf) or a
    string like "12.5", "-3", "1e-3", "1,200.50", "5/36", "$12", "12 EUR", "12.5%" (the % is
    only stripped). "about 12", "12 kg", "12,5" raise ValueError."""
    if hasattr(x, "item") and not isinstance(x, (list, str)):  # numpy scalar
        x = x.item()
    if isinstance(x, bool):
        raise ValueError("a boolean is not a number")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        s = x.strip().replace("\u2212", "-")  # Unicode minus
        sign = ""
        if s[:1] in "+-" and len(s) > 1:
            sign, s = s[0], s[1:].strip()
        if s.upper().endswith("EUR"):
            s = s[:-3].strip()
        if s[:1] in _CURRENCY:
            s = s[1:].strip()
        elif s[-1:] in _CURRENCY:
            s = s[:-1].strip()
        s = sign + s.removesuffix("%").strip()
        if _PLAIN.match(s):
            v = float(s)
        elif _THOUSANDS.match(s):
            v = float(s.replace(",", ""))
        elif (m := _FRACTION.match(s)) and int(m.group(2)) != 0:
            v = int(m.group(1)) / int(m.group(2))
        else:
            raise ValueError(f"cannot read a number from {x!r}")
    else:
        raise ValueError(f"expected a number or a string, got {type(x).__name__}")
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {x!r}")
    return v


def coerce_label(x) -> str:
    """Accept what the scorer accepts as a label: a string (returned stripped, case kept) or
    an integer (returned as its decimal string). The scorer compares after casefold()."""
    if hasattr(x, "item") and not isinstance(x, (list, str)):
        x = x.item()
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    raise ValueError(f"expected a string label, got {type(x).__name__}")


def coerce_list(x, length: int | None = None) -> list:
    """A list, tuple or 1-D numpy array -> list (optionally of exactly `length` items)."""
    if hasattr(x, "ndim"):
        if x.ndim != 1:
            raise ValueError(f"expected a 1-D array, got shape {x.shape}")
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"expected a list, got {type(x).__name__}")
    if length is not None and len(x) != length:
        raise ValueError(f"expected {length} values, got {len(x)}")
    return list(x)


def to_answer(value, answer_type: str):
    """Turn a Python value (e.g. RESULT from run_python) into what the scorer accepts for
    `answer_type` (number, category, list, vector, predictions), or raise ValueError.
    Lengths are not checked: only the prompt says how many values are expected."""
    if answer_type == "number":
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        return coerce_number(value)
    if answer_type == "category":
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        return coerce_label(value)
    if answer_type == "list":
        return [coerce_label(v) for v in coerce_list(value)]
    if answer_type in ("vector", "predictions"):
        value = coerce_list(value)
        if answer_type == "vector":
            return [coerce_number(v) for v in value]
        # Keep str and int as they are: the scorer reads them as labels (macro-F1) or as
        # numbers (regression). A float label like 1.0 would fail as a class label.
        return [v if isinstance(v, (str, int)) and not isinstance(v, bool) else coerce_number(v)
                for v in value]
    raise ValueError(f"unknown answer_type {answer_type!r}")


# --- tolerant parsing of free text (the model's reply) ------------------------------------

_MARK = re.compile(r"####\s*(.+)")
_NUM_TOKEN = re.compile(r"[+-]?\d+\s*/\s*\d+|[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+")


def answer_tail(text: str) -> str:
    """The text after the LAST '#### ' marker, or the whole text when there is none."""
    found = _MARK.findall(text)
    return found[-1].strip() if found else text.strip()


def parse_number(text: str) -> float | None:
    """Read a number from a reply: first the whole tail after '####' under the scorer's
    rules, else the last number-looking token in it (then in the whole reply). None if none."""
    text = text.replace("\u2212", "-")
    tail = answer_tail(text).strip("`*_ .")
    try:
        return coerce_number(tail)
    except ValueError:
        pass
    for source in (tail, text):
        for tok in reversed(_NUM_TOKEN.findall(source)):
            try:
                return coerce_number(tok.replace(" ", ""))
            except ValueError:
                continue
    return None


def parse_label(text: str, choices: list[str] | None = None) -> str | None:
    """Read a label from a reply. With `choices` (e.g. the distinct values of a CSV column),
    return the choice mentioned last in the tail (case-insensitive), else None."""
    tail = answer_tail(text).strip().strip("`*_\"'. ").strip()
    if choices is None:
        return tail or None
    low = tail.casefold()
    exact = [c for c in choices if c.casefold() == low]
    if exact:
        return exact[0]
    hits = [(low.rfind(c.casefold()), c) for c in choices if c.casefold() in low]
    return max(hits)[1] if hits else None


def parse_list(text: str, kind: str = "number") -> list:
    """Read a comma / semicolon / newline separated list from the tail of a reply.
    kind="number" keeps only readable numbers; kind="label" keeps stripped strings.
    Commas separate items here, so "1,200.5" reads as [1, 200.5]: tell your model not to
    write thousands separators in lists."""
    tail = answer_tail(text).strip().strip("[]()`")
    items = [s.strip().strip("[]()'\"` ") for s in re.split(r"[,;\n]\s*(?=\S)", tail)]
    if kind == "label":
        return [s for s in items if s]
    out = []
    for s in items:
        v = parse_number(s)
        if v is not None:
            out.append(v)
    return out


# ============================================================================ budget


class Budget:
    """A stopwatch against a time budget.

    Each task carries `time_budget_s`, its share of the call's timeout, so a batch's
    budgets add up to that timeout. Missing it is the worst outcome: the run ends and the
    platform fails the deployment, with no score. So keep a margin (`safety`) and check
    `remaining()` before each expensive step.
    """

    def __init__(self, seconds: float, safety: float = 0.8):
        self.total = seconds * safety
        self.t0 = time.monotonic()

    @classmethod
    def for_tasks(cls, tasks: list[dict], safety: float = 0.8) -> "Budget":
        return cls(sum(t["time_budget_s"] for t in tasks), safety)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def remaining(self) -> float:
        return self.total - self.elapsed()

    def has(self, seconds: float) -> bool:
        """True if `seconds` more work still fits in the budget."""
        return self.remaining() > seconds

"""dsh -- the DS-Harness starter kit (one file: upload it next to your agent.py).

What it gives you, the plumbing that is fiddly to get right:

    load_llm(model_id)        a small chat model with batched generation (`LLM.chat`)
    ALLOWED_MODELS            the models the platform mounts for this challenge
    tagged_number(text)       the number on the last "ANSWER: ..." line of a reply, or None
    last_number(text)         the last number written in a reply, or None
    to_number(x)              exactly what the scorer accepts as an answer (else ValueError)
    calculator(expr)          a safe arithmetic evaluator: the first tool
    parse_tool_call(text)     the first <call tool="...">...</call> in a reply, or None
    run_python(code, files)   run code in a subprocess next to the task's files, read RESULT
    file_preview(paths)       the first lines of each file, as text
    Clock                     a stopwatch against the 40 s of a solve() call

What it does not give you, which is your project: recognising the kind of task,
reading the files and deciding what reaches the model, which tools exist and
how they are called, checking the answers, spending the 40 s well.

Platform facts this file is written for: no network (models come from a
read-only cache), a read-only root filesystem, /tmp is a 128 MB tmpfs, 3 CPUs
and 3 GiB of RAM for the agent container, one RTX 4090 for your job.
"""
from __future__ import annotations

import ast
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

ALLOWED_MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen3-1.7B",
)
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ============================================================================ LLM


class LLM:
    """A causal LM and its tokenizer, with one method: `chat`."""

    def __init__(self, model, tokenizer, device: str):
        self.model, self.tok, self.device = model, tokenizer, device
        # Qwen3 templates accept `enable_thinking`; thinking is OFF by default here:
        # thousands of thinking tokens per task do not fit in 40 s for 8 tasks.
        self.has_thinking_flag = "enable_thinking" in (tokenizer.chat_template or "")

    def chat(self, conversations: list[list[dict]], max_new_tokens: int = 512,
             temperature: float = 0.0, batch_size: int = 8, thinking: bool = False,
             max_time: float | None = None, stop: list[str] | None = None) -> list[str]:
        """One reply per conversation (a list of {"role", "content"} messages), in order.

        Greedy decoding when temperature == 0. `max_time` bounds the whole call in seconds:
        generation stops when it runs out (replies are cut, sub-batches not started come
        back as ""). A generate() call cannot be interrupted from outside, so this is how
        you keep it inside the 40 s of a solve() call. `stop`: strings that end a reply
        early (e.g. ["</call>"] to hand control back to your code after a tool call); the
        stop string stays at the end of the reply.
        """
        import torch

        kw = {"enable_thinking": thinking} if self.has_thinking_flag else {}
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
                 for m in conversations]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding per sub-batch
        replies = [""] * len(texts)
        gen = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature > 0:
            gen.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen.update(do_sample=False)
        if stop:
            gen.update(stop_strings=list(stop), tokenizer=self.tok)
        t_end = None if max_time is None else time.monotonic() + max_time
        for k in range(0, len(order), batch_size):
            idx = order[k:k + batch_size]
            if t_end is not None:
                left = t_end - time.monotonic()
                if left <= 0.5:
                    break
                gen["max_time"] = left
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen)
            new = out[:, enc["input_ids"].shape[1]:]
            for i, text in zip(idx, self.tok.batch_decode(new, skip_special_tokens=True)):
                replies[i] = text
        return replies


def cached_revision(model_id: str) -> str | None:
    """The commit of `model_id` in the local HF cache, or None when it is not cached.

    The platform's upload scan refuses a `from_pretrained` without a pinned `revision`;
    pinning the cached snapshot satisfies it and loads exactly the platform's weights.
    On Colab a model not cached yet gives None: it is downloaded on first use."""
    from huggingface_hub import constants

    ref = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"), "refs", "main")
    if not os.path.exists(ref):
        return None
    with open(ref) as f:
        return f.read().strip()


def load_llm(model_id: str = DEFAULT_MODEL, device: str | None = None) -> LLM:
    """Load a chat model: fp16 on a GPU, straight onto the device (the agent container has
    3 GiB of RAM, so the weights must not pass through it in full)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id not in ALLOWED_MODELS:
        raise ValueError(f"{model_id} is not mounted on the platform; choose one of {ALLOWED_MODELS}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    revision = cached_revision(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"  # decoder-only models continue after the last token: pad on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 device_map=device)
    model.eval()
    return LLM(model, tok, device)


# ============================================================================ answers

_PLAIN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+")


def to_number(x) -> float:
    """What the scorer accepts: an int or float (not bool, not NaN/inf), or a string that
    is a plain decimal number ("41.07", "-3", "1e-3"). "1,200", "12 kg", "about 3" raise."""
    if hasattr(x, "item") and not isinstance(x, (list, str)):  # numpy scalar
        x = x.item()
    if isinstance(x, bool):
        raise ValueError("a boolean is not a number")
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str) and _PLAIN.match(x.strip()):
        v = float(x.strip())
    else:
        raise ValueError(f"not a plain number: {str(x)[:40]!r}")
    if not math.isfinite(v):
        raise ValueError(f"non-finite number {x!r}")
    return v


def _read(token: str) -> float | None:
    try:
        return to_number(token.replace(",", "").replace("−", "-"))
    except ValueError:
        return None


def tagged_number(text: str, tag: str = "ANSWER:") -> float | None:
    """The first number on the last line that starts with `tag`, or None."""
    lines = [ln.strip().strip("*`") for ln in text.splitlines()]
    for ln in reversed(lines):
        if ln.upper().startswith(tag.upper()):
            found = _NUMBER.findall(ln[len(tag):])
            return _read(found[0]) if found else None
    return None


def last_number(text: str) -> float | None:
    """The last number written anywhere in the text ("1,200.5" reads 1200.5), or None."""
    found = _NUMBER.findall(text.replace("−", "-"))
    return _read(found[-1]) if found else None


# ============================================================================ tools

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in (
    "sqrt", "exp", "log", "log10", "log2", "floor", "ceil", "factorial", "comb", "perm", "fabs")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def calculator(expr: str) -> float:
    """Evaluate an arithmetic expression: numbers, + - * / // % **, parentheses, pi, e and
    sqrt exp log log10 log2 floor ceil factorial comb perm fabs abs round min max.
    No names, attributes or imports. Raises ValueError on anything else."""
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
            raise ValueError("'^' is not a power: use **")
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

    expr = expr.strip().strip("`").replace("×", "*").replace("÷", "/").replace("−", "-")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse {expr!r}: {e.msg}") from e
    return float(ev(tree))


_CALL = re.compile(r"<call\s+tool\s*=\s*[\"']?([\w.-]+)[\"']?\s*>(.*?)</call>", re.S)


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """The first `<call tool="name">argument</call>` in a reply, as (name, argument)."""
    m = _CALL.search(text)
    return (m.group(1), m.group(2).strip()) if m else None


# ============================================================================ code


@dataclass
class ExecResult:
    ok: bool            # the code ran and defined RESULT
    value: object       # RESULT, as plain Python (numpy/pandas values converted)
    stdout: str         # what the code printed (truncated)
    error: str | None   # traceback tail, timeout or "no RESULT": send it back to the model
    seconds: float


_RUNNER = r"""
import json, sys, traceback
ns = {"__name__": "__main__"}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
except BaseException:
    traceback.print_exc()
    sys.exit(1)
if "RESULT" not in ns:
    sys.stderr.write("NameError: the code must assign its answer to RESULT\n")
    sys.exit(1)
def plain(x):
    if hasattr(x, "tolist"):
        return plain(x.tolist())
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x
try:
    json.dump(plain(ns["RESULT"]), open("result.json", "w"), allow_nan=True)
except (TypeError, ValueError) as e:
    sys.stderr.write(f"TypeError: RESULT must be a number ({e})\n")
    sys.exit(1)
"""


def run_python(code: str, files: list[str] = (), timeout_s: float = 10.0,
               mem_limit_mb: int = 1024, max_output: int = 2000) -> ExecResult:
    """Run `code` in a fresh Python subprocess whose working directory holds the task's
    files (linked by their base name, so `pd.read_csv("loans.csv")` works). The code must
    assign its answer to RESULT. A cold Python + pandas start costs about 0.5-1 s."""
    t0 = time.monotonic()
    d = tempfile.mkdtemp(prefix="dsh_")
    try:
        for p in files:
            os.symlink(os.path.abspath(p), os.path.join(d, os.path.basename(p)))
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "_runner.py"), "w") as f:
            f.write(_RUNNER)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")

        def limit():
            if sys.platform.startswith("linux"):
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit_mb << 20, mem_limit_mb << 20))

        p = subprocess.Popen([sys.executable, "_runner.py"], cwd=d, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, preexec_fn=limit)
        try:
            out, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return ExecResult(False, None, out[-max_output:], f"TimeoutError: over {timeout_s:.0f} s",
                              time.monotonic() - t0)
        if p.returncode != 0:
            return ExecResult(False, None, out[-max_output:], err[-max_output:] or f"exit {p.returncode}",
                              time.monotonic() - t0)
        with open(os.path.join(d, "result.json")) as f:
            value = json.load(f)
        return ExecResult(True, value, out[-max_output:], None, time.monotonic() - t0)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def file_preview(paths: list[str], max_lines: int = 6) -> str:
    """The first `max_lines` lines of each file, under its name."""
    parts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            lines = [next(f, "") for _ in range(max_lines)]
        n = sum(1 for _ in open(p, encoding="utf-8"))
        parts.append(f"--- {os.path.basename(p)} ({n} lines) ---\n" + "".join(lines).rstrip())
    return "\n".join(parts)


# ============================================================================ time


class Clock:
    """A stopwatch against the time of one solve() call (40 s on the platform).
    A call that runs out of time ends your run: keep a margin."""

    def __init__(self, seconds: float = 40.0, margin: float = 6.0):
        self.t0, self.limit = time.monotonic(), seconds - margin

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def left(self) -> float:
        return self.limit - self.elapsed()

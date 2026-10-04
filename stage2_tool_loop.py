"""Stage 2 -- the model calls a calculator, in a loop (a skeleton: the loop body is yours).

Upload as agent.py with dsh.py, PyTorch runtime, once the TODO is written. The parts:

    tool spec     in the system prompt: what the calculator accepts and how to call it
    call format   <call tool="calculator">EXPRESSION</call>, shown once in a worked exchange
                  (DEMO); generation stops at </call>
    call parser   dsh.parse_tool_call(reply) -> (name, argument) or None
    execution     observe(call): dsh.calculator, an error returned as text
    observation   sent back to the model as a user message: "Result: ..." or "Error: ..."
    stop rule     a reply without a call, MAX_ROUNDS rounds, or the clock

Test it on the calc.* types against stage1_direct.py:
    python localtest.py stage2_tool_loop.py --type calc.arithmetic --type calc.functions ...
"""
import os

import dsh

MODEL = os.environ.get("DSH_MODEL", dsh.DEFAULT_MODEL)
MAX_ROUNDS = 4
SYSTEM = (
    "Solve the problem step by step. You have a calculator: to use it, write\n"
    '<call tool="calculator">EXPRESSION</call>\n'
    "and stop; the result comes back in the next message. EXPRESSION is Python arithmetic: "
    "numbers, + - * / ** // %, parentheses, and the functions sqrt, exp, log (natural), log10, "
    "comb, perm, factorial, floor, ceil, round. Use the calculator for every computation, "
    "one call at a time. When you know the final answer, write a last line of the form "
    "ANSWER: <number>, rounded as the question asks."
)
# One worked exchange, shown before the task: a small model follows a call format it has
# seen used far better than one it has only read about (measured: 9 -> 41 on the tasks
# without files, Qwen2.5-1.5B).
DEMO = [
    {"role": "user", "content": "A jacket costs 80 euros. What does it cost after a 15% discount, "
                                "then a 20% tax? Round to 2 decimals."},
    {"role": "assistant", "content": "The discounted price is 80 × (1 − 0.15), then the tax multiplies "
                                     "it by 1.20.\n<call tool=\"calculator\">80 * (1 - 0.15) * 1.20</call>"},
    {"role": "user", "content": "Result: 81.6"},
    {"role": "assistant", "content": "The final price is 81.6 euros.\nANSWER: 81.6"},
]


def observe(call):
    """Run one tool call; always return text for the model, never raise."""
    name, arg = call
    if name != "calculator":
        return f"Error: unknown tool {name!r}; the only tool is calculator."
    try:
        v = dsh.calculator(arg)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f"Error: {e}"
    return f"Result: {v:.10g}"


class Agent:
    def __init__(self):
        self.llm = dsh.load_llm(MODEL)

    def solve(self, tasks, seconds=40.0):
        # seconds: this call's time; less when another part of your system shares the call
        clock = dsh.Clock(seconds)
        convs = []
        for t in tasks:
            text = t["objective"]
            if t["files"]:
                text += "\n\nThe files (first lines):\n" + dsh.file_preview(t["files"])
            convs.append([{"role": "system", "content": SYSTEM}] + DEMO + [{"role": "user", "content": text}])
        final = [""] * len(tasks)        # the last reply of each conversation
        active = list(range(len(tasks)))  # conversations still waiting for a tool result
        for _ in range(MAX_ROUNDS):
            if not active or clock.left() < 3:
                break
            replies = self.llm.chat([convs[i] for i in active], max_new_tokens=384,
                                    max_time=clock.left() - 1, stop=["</call>"])
            # TODO -- the loop body. For each conversation i of `active` and its reply r:
            #   1. keep r as final[i];
            #   2. parse a tool call from r (dsh.parse_tool_call); no call: conversation i is done;
            #   3. otherwise append r as the assistant's message and observe(call) as a user
            #      message to convs[i], and keep i active for the next round.
            # Then set `active` to the conversations still waiting.
            raise NotImplementedError("write the loop body")
        out = []
        for t, r in zip(tasks, final):
            answer = dsh.tagged_number(r)
            if answer is None:
                answer = dsh.last_number(r)
            out.append({"id": t["id"], "answer": answer if answer is not None else 0})
        return out

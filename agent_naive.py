"""Naive baseline (published): the model answers directly, no tools, no code execution.

It reads the question and an overview of each CSV, reasons briefly, and ends with a line
"#### <answer>". It is weak on purpose: a 1.5B model cannot compute a variance over
800 rows or fit a classifier in its head. Your harness must beat it everywhere.

This is grading anchor A0. To submit: upload this file renamed to agent.py, plus dsh.py.
"""
import os
import traceback

import dsh

# Local override for quick tests (e.g. DSH_MODEL=Qwen/Qwen2.5-0.5B-Instruct); on the
# platform the constant is used.
MODEL_ID = os.environ.get("DSH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Small models copy the format of the example they see, literally. An abstract
# placeholder ("#### ANSWER") was copied as is, so every format is shown with real values.
SYSTEM = (
    "You are a careful data scientist. Think step by step, briefly. Then write a last line "
    "that starts with #### followed by the answer only.\n"
    "Examples of last lines:\n"
    "#### 42.5\n"
    "#### 0.1667\n"
    "#### north\n"
    "#### Lyon, Nice, Paris\n"
    "#### 12.1, 13.4, 15.0\n"
    "For a number write only the number: no units, no words."
)
FORMAT_HINT = {
    "number": "The answer is one number, e.g. `#### 42.5`.",
    "category": "The answer is one label taken from the data, e.g. `#### north`.",
    "list": "The answer is a comma-separated list of labels, e.g. `#### Lyon, Nice, Paris`.",
    "vector": "The answer is a comma-separated list of numbers, e.g. `#### 12.1, 13.4, 15.0`.",
    "predictions": "The answer is a comma-separated list, one value per requested test row, "
                   "e.g. `#### yes, no, no` or `#### 31.2, 18.0, 25.5`.",
}
# Well-typed placeholder when parsing fails: it scores 0 but is not a format error.
PLACEHOLDER = {"number": 0.0, "category": "", "list": [], "vector": [], "predictions": []}


class Agent:
    def __init__(self):
        self.llm = dsh.load_llm(MODEL_ID)

    def solve(self, tasks):
        # On the platform, an exception escaping solve() ENDS THE RUN and the deployment
        # fails with no score. So every step that can fail is caught here, per task.
        budget = dsh.Budget.for_tasks(tasks)
        msgs, notes = [], []
        for t in tasks:
            try:
                msgs.append(self.messages(t))
                notes.append("")
            except Exception:
                msgs.append(None)
                notes.append(traceback.format_exc(limit=2))
        todo = [i for i, m in enumerate(msgs) if m is not None]
        replies = [""] * len(tasks)
        try:
            for i, r in zip(todo, self.llm.chat([msgs[i] for i in todo], max_new_tokens=384,
                                                max_time=budget.remaining())):
                replies[i] = r
        except Exception:  # e.g. out of GPU memory: answer every task with the placeholder
            notes = [n + traceback.format_exc(limit=2) for n in notes]
        out = []
        for t, r, note in zip(tasks, replies, notes):
            try:
                answer = self.parse(t, r)
            except Exception:
                answer, note = PLACEHOLDER[t["answer_type"]], note + traceback.format_exc(limit=2)
            out.append({"id": t["id"], "answer": answer, "trace": r + ("\n" + note if note else "")})
        return out

    @staticmethod
    def messages(task):
        user = task["prompt"]
        if task["files"]:
            user += "\n\n" + dsh.csv_overview(task["files"], n_rows=5)
        user += "\n\n" + FORMAT_HINT[task["answer_type"]]
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def parse(task, reply):
        kind = task["answer_type"]
        if kind == "number":
            v = dsh.parse_number(reply)
            return 0.0 if v is None else v
        if kind == "category":
            return dsh.parse_label(reply) or ""
        if kind == "vector":
            return dsh.parse_list(reply, "number")
        # list and predictions: labels as written (numbers stay strings like "31.2", which
        # the scorer also reads as numbers). A wrong length is left for the scorer to judge.
        return dsh.parse_list(reply, "label")

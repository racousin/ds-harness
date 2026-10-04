"""Stage 1 -- the model answers directly (the notebook's section 3, in one file).

Upload as agent.py with dsh.py, PyTorch runtime. One model call per task, all 8
tasks of a call in one batch:

    system prompt  reason step by step, then a last line "ANSWER: <number>"
    user message   the objective (and, for a task with files, their first lines)
    parser         the ANSWER line, else the last number of the reply, else 0

What it shows: a small model with a well-parsed answer gets part of the
computation tasks; with files it hits a wall -- it never sees the data.
"""
import os

import dsh

MODEL = os.environ.get("DSH_MODEL", dsh.DEFAULT_MODEL)
SYSTEM = ("Solve the problem. Reason step by step, writing each calculation. "
          "Then write a last line of the form ANSWER: <number>, with the number only.")


class Agent:
    def __init__(self):
        self.llm = dsh.load_llm(MODEL)

    def prompt(self, task):
        text = task["objective"]
        if task["files"]:
            text += "\n\nThe files (first lines):\n" + dsh.file_preview(task["files"])
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]

    def solve(self, tasks, seconds=40.0):
        # seconds: this call's time; less when another part of your system shares the call
        clock = dsh.Clock(seconds)
        try:
            replies = self.llm.chat([self.prompt(t) for t in tasks], max_new_tokens=512,
                                    max_time=clock.left())
        except Exception as e:  # never let an exception end the run: answer 0 instead
            print(f"generation failed: {e!r}")
            replies = [""] * len(tasks)
        out = []
        for t, r in zip(tasks, replies):
            answer = dsh.tagged_number(r)
            if answer is None:
                answer = dsh.last_number(r)
            out.append({"id": t["id"], "answer": answer if answer is not None else 0})
        return out

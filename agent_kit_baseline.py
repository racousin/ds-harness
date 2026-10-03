"""Kit baseline: the simplest program-of-thought harness built from dsh.

For every task the model writes one Python block that sets RESULT; dsh.run_python runs it
with the task's CSVs. A failed attempt gets ONE repair round with the error message. If that
fails too, a well-typed placeholder answers (it scores 0, but the run goes on).

This is grading anchor A1, not a strong agent. Obvious next steps, all yours:
  * routing: a calculator or a direct answer for pure maths, code only for tables
  * few-shot banks per task family (a worked example beats any instruction)
  * self-consistency: sample k programs, vote on RESULT
  * verification: sanity checks (a probability in [0, 1], a count is an integer, the
    number of predictions equals the number of rows the prompt asks for, ...)
  * custom tools: fit_and_predict(train, test, target, drop), forecast(series, h), ...
  * better fallbacks than a placeholder, lenient RESULT capture in your own runner

To submit: upload this file renamed to agent.py, plus dsh.py.
"""
import os
import re
import traceback

import dsh

MODEL_ID = os.environ.get("DSH_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CODE_TIMEOUT_S = 20
RUN_S = 1.5  # time kept per task for running its code (python + pandas start ~0.5-1 s)

SYSTEM = (
    "You solve data tasks by writing Python. Reply with ONE ```python code block and nothing "
    "else. The CSV files are in the current directory (read them with pd.read_csv('name.csv')). "
    "Available: pandas, numpy, math. NOT available: scikit-learn (sklearn), scipy, "
    "statsmodels: never import them. "
    "Follow every rule in the question: which rows and columns to use, how missing or messy "
    "values are written, units and rounding. Store the final answer in a variable named RESULT "
    "(uppercase), for example `RESULT = 42.5`."
)
GUIDE = {
    "number": "RESULT must be one Python number, e.g. `RESULT = round(df['price'].mean(), 2)`. "
              "For word problems, compute every step in code.",
    "category": "RESULT must be one string exactly as written in the data, e.g. `RESULT = 'north'`.",
    "list": "RESULT must be a list of strings, in the order asked, e.g. `RESULT = ['Lyon', 'Nice']`.",
    "vector": "RESULT must be a list of floats with the requested length, "
              "e.g. `RESULT = [float(v) for v in forecast]`.",
    "predictions": "RESULT must be a list with one value per requested row of test.csv, in file "
                   "order, e.g. `RESULT = list(pred)`, computed with numpy and pandas only. Use "
                   "only the feature columns allowed by the question, fill missing values "
                   "(fillna), encode text columns with pd.get_dummies.",
}
# Well-typed placeholder: it scores 0 but keeps the reply valid.
PLACEHOLDER = {"number": 0.0, "category": "", "list": [], "vector": [], "predictions": []}
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(reply):
    """The first fenced block; an unfenced reply is taken as code (small models do that)."""
    blocks = _CODE.findall(reply)
    return blocks[0] if blocks else reply.replace("```python", "").replace("```", "")


class Agent:
    def __init__(self):
        self.llm = dsh.load_llm(MODEL_ID)

    def solve(self, tasks):
        # On the platform, an exception escaping solve() ENDS THE RUN and the deployment
        # fails with no score, so nothing may escape: failures are caught per task and step.
        budget = dsh.Budget.for_tasks(tasks)
        n = len(tasks)
        answers = [PLACEHOLDER.get(t["answer_type"], 0.0) for t in tasks]
        traces = [""] * n
        msgs = [self.safe(self.messages, t) for t in tasks]
        todo = [i for i in range(n) if isinstance(msgs[i], list)]
        for i in range(n):
            if i not in todo:
                traces[i] = f"# prompt failed\n{msgs[i]}"
        # Attempt 1 (batched), then one repair round (batched) for the failures.
        for attempt in (1, 2):
            if not todo or not budget.has(RUN_S * len(todo) + 2):
                break
            try:
                replies = self.llm.chat([msgs[i] for i in todo], max_new_tokens=400,
                                        max_time=budget.remaining() - RUN_S * len(todo))
            except Exception:
                for i in todo:
                    traces[i] += f"\n# attempt {attempt}: generation failed\n{traceback.format_exc(limit=2)}"
                break
            failed = []
            for i, reply in zip(todo, replies):
                ok, value = self.safe(self.run, tasks[i], reply, budget)
                traces[i] += f"\n# attempt {attempt}\n{extract_code(reply)}\n# -> {str(value)[-600:]}"
                if ok:
                    answers[i] = value
                else:
                    failed.append(i)
                    msgs[i] = msgs[i] + [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": f"The code failed:\n{str(value)[-800:]}\n"
                                                    "Fix it. Reply with ONE python code block that sets RESULT."}]
            todo = failed
        return [{"id": t["id"], "answer": a, "trace": tr} for t, a, tr in zip(tasks, answers, traces)]

    @staticmethod
    def safe(fn, *args):
        """Call fn; on any exception return (False, traceback) instead of raising."""
        try:
            return fn(*args)
        except Exception:
            return False, traceback.format_exc(limit=3)

    @staticmethod
    def messages(task):
        user = task["prompt"]
        if task["files"]:
            user += "\n\n" + dsh.csv_overview(task["files"])
        user += "\n\n" + GUIDE[task["answer_type"]]
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    @staticmethod
    def run(task, reply, budget):
        """Execute the code; return (True, answer) or (False, error message)."""
        if not budget.has(2):
            return False, "out of time budget"
        res = dsh.run_python(extract_code(reply), task["files"],
                             timeout_s=min(CODE_TIMEOUT_S, budget.remaining()))
        if not res.ok:
            return False, res.error
        try:
            return True, dsh.to_answer(res.value, task["answer_type"])
        except ValueError as e:
            return False, f"RESULT has the wrong type: {e}"

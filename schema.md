# DS-Harness — task and answer format

`dev.json` (178 tasks, downloaded from the challenge page) is a list of tasks with their gold
answers. The platform sends your agent the same kind of task, without the gold keys, and scores
the replies with `scoring.py` (shipped here unchanged). `localtest.py` runs your agent on
`dev.json` with the platform's delivery, timeouts and scoring.

## Agent interface

```python
class Agent:
    def __init__(self): ...                       # load your model here (60 s on the platform)
    def solve(self, tasks: list[dict]) -> list[dict]:
        return [{"id": t["id"], "answer": ..., "trace": "optional text"} for t in tasks]
```

Each task you receive:

| key | content |
|---|---|
| `id` | string, echo it back |
| `prompt` | the question, in English prose |
| `files` | `{file name: CSV text}`; may be empty. Often includes `data_dictionary.csv` (columns `file`, `column`, `description`, `unit`) |
| `answer_type` | `number`, `category`, `list`, `vector` or `predictions` |
| `time_budget_s` | this task's share of the call's timeout: stop working on the task by then |

`dev.json` also carries `family`, `level`, `answer`, `scoring` and `heldout_family` so you can
score yourself; the agent never receives them.

## Answer types

| `answer_type` | what to return | example |
|---|---|---|
| `number` | an int/float (numpy scalars accepted locally), or a string such as `"12.5"`, `"1,200.50"`, `"-3"`, `"5/36"`, `"12.5%"` (read as 12.5) | `41.07` |
| `category` | a string, compared case-insensitively after stripping spaces | `"CAR"` |
| `list` | a list of labels, compared position by position (case-insensitive) | `["Lyon", "Nice"]` |
| `vector` | a list of numbers of the requested length (a forecast) | `[102.4, 98.1, 110.0]` |
| `predictions` | a list with one element per requested test row: labels or numbers, as the prompt says (probabilities are numbers in [0, 1]) | `["yes", "no", ...]` |

Rejected (counted as format errors, score 0): booleans, NaN/inf, `"12,5"` (ambiguous), text
around a number (`"about 12"`), a list of the wrong length, a probability outside [0, 1].
In a reply, an id that is not in the batch or an id given twice counts as a format error;
a duplicated id also scores 0.

The prompt fixes the rounding, units, ddof, quartile method, inclusive bounds and how to treat
missing or messy values. Tolerances are in each task's `scoring`.

## Scoring

- Exact tasks: 1 if within tolerance, else 0.
- Modelling tasks (level 3): `(m - baseline) / (reference - baseline)` clipped to [0, 1], where
  `m` is sMAPE (forecast lists), absolute percentage error (a forecast total), MAE (regression),
  macro-F1 (labels) or Brier score (probabilities). The baseline is a trivial rule (seasonal naive,
  training median, training prior, best of majority/random labels); the reference is a tuned model.
- Level score = mean score over all tasks of that level (every task of a level weighs the same).
- **Leaderboard score = 100 × (0.3·L1 + 0.4·L2 + 0.3·L3).**
- Extra leaderboard columns: `level_1`, `level_2`, `level_3`, `heldout` (mean over the tasks of
  families absent from the public tasks), `format_errors`, `tasks_answered`.

## Delivery and time

- The private run makes 21 `solve` calls: 8 level-1/2 tasks per call, or 2 level-3 tasks.
- Call timeout = 2.5 × (2 s per level-1/2 task + 8 s per level-3 task): 40 s for a full batch.
  Each task's `time_budget_s` is its share of that timeout.
- The job ends 399 s after it starts, model loading included (about 380 s of answering after a
  ~17 s load). Every call gets its full timeout: when the next batch's full timeout no longer
  fits before the end, that batch is not sent and its tasks score 0. **Plan for 330 s of
  answering**, about 2 s per level-1/2 task and 8 s per level-3 task.
- **A call that raises or misses its timeout ends the run, and the deployment fails: no
  score.** Catch errors per task, return a placeholder answer instead of raising, and stop at
  `time_budget_s`.

## Data notes

- Column headers and data-dictionary descriptions vary from task to task, even for the same
  kind of table. Read the dictionary (or the column list in the prompt), never a fixed header.
- In level-3 tasks, feature columns of `train.csv` and `test.csv` may contain missing values
  (empty cells) without any mention in the prompt. Impute or handle them: an exception in
  `solve` ends the run with no score. Columns used in a row condition of the prompt have no missing values.
- In level-1/2 table tasks, when the prompt says how missing values are written, rows missing a
  column are left out of every computation that uses that column (including the denominator of
  a percentage).

## Private set

119 tasks (37 at level 1, 67 at level 2, 15 at level 3) with the same format. It uses other
wordings, other table domains and column names. 24 of its tasks come from families that are
not in `dev.json`; the `heldout` column is their mean. `dev.json` contains one unseen-style
family, `sample.cumulative_first` (2 tasks, `heldout_family` true), as a preview; it does not
appear in the private set. A harness that reads the prompt and the files generalises. A solver
keyed on dev wordings does not.

`localtest.py --budget-s 330` scales the 330 s to the task file you run (about 575 s for the
full `dev.json`, which has more level-3 tasks than the private set). Expect dev scores to
run higher than private ones: you tune on the dev wordings.

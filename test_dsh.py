"""Unit tests for dsh (no LLM needed): pytest test_dsh.py"""
import glob
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsh  # noqa: E402

CSV = {"sales.csv": "region,amount\nnorth,10\nsouth,20\nnorth,30\n"}
ROOT = tempfile.gettempdir()


def _leftovers():
    return set(glob.glob(os.path.join(ROOT, "dsh_*")))


# --- run_python -------------------------------------------------------------------------

def test_run_python_reads_csv_and_returns_result():
    before = _leftovers()
    r = dsh.run_python("import pandas as pd\ndf = pd.read_csv('sales.csv')\n"
                       "print('hi')\nRESULT = df.groupby('region').amount.mean()['north']",
                       CSV, workdir_root=ROOT)
    assert r.ok and r.value == 20.0 and r.error is None and "hi" in r.stdout
    assert _leftovers() == before  # temp dir cleaned


def test_run_python_numpy_values_become_plain():
    r = dsh.run_python("import numpy as np\nRESULT = np.arange(3) * 1.5", {}, workdir_root=ROOT)
    assert r.ok and r.value == [0.0, 1.5, 3.0]


def test_run_python_missing_result_hints_the_name():
    r = dsh.run_python("result = 3", {}, workdir_root=ROOT)
    assert not r.ok and "RESULT" in r.error and "'result'" in r.error


def test_run_python_traceback_and_none():
    r = dsh.run_python("import statsmodels", {}, workdir_root=ROOT)
    assert not r.ok and "ModuleNotFoundError" in r.error
    r = dsh.run_python("RESULT = None", {}, workdir_root=ROOT)
    assert not r.ok and "None" in r.error


def test_run_python_timeout_and_cleanup():
    before = _leftovers()
    r = dsh.run_python("while True: pass", {}, timeout_s=1, workdir_root=ROOT)
    assert not r.ok and "TimeoutError" in r.error and r.seconds < 5
    assert _leftovers() == before


def test_run_python_non_json_result():
    r = dsh.run_python("RESULT = object()", {}, workdir_root=ROOT)
    assert not r.ok and "TypeError" in r.error


def test_run_python_rejects_paths():
    with pytest.raises(ValueError):
        dsh.run_python("RESULT = 1", {"../evil.csv": "x"}, workdir_root=ROOT)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is Linux-only")
def test_run_python_memory_limit():
    r = dsh.run_python("x = bytearray(2 * 1024**3)\nRESULT = 1", {}, mem_limit_mb=512,
                       workdir_root=ROOT)
    assert not r.ok and "MemoryError" in r.error


# --- calculator -------------------------------------------------------------------------

@pytest.mark.parametrize("expr,val", [
    ("2 + 3 * 4", 14), ("(1 + 0.05) ** 10", 1.05 ** 10), ("-7 // 2", -4), ("7 % 3", 1),
    ("sqrt(16) + log(e)", 5), ("comb(5, 2) / 36", 10 / 36), ("round(2 * pi, 2)", 6.28),
])
def test_calculator(expr, val):
    assert dsh.calculator(expr) == pytest.approx(val)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "2 ^ 3", "9 ** 99999",
                                  "(1).__class__", "x + 1", "2 +"])
def test_calculator_rejects(expr):
    with pytest.raises(ValueError):
        dsh.calculator(expr)


# --- csv_overview -----------------------------------------------------------------------

def test_csv_overview():
    txt = dsh.csv_overview({"a.csv": "x,y\n1,\n2,b\n3,c\n4,d\n"}, n_rows=2, show_all_below=3)
    assert "4 rows x 2 columns" in txt and "x (int64)" in txt and "1 missing" in txt
    assert "3," not in txt  # only 2 rows shown


# --- coercion mirrors the scorer ----------------------------------------------------------

@pytest.mark.parametrize("raw,val", [
    ("12.5", 12.5), (" -3 ", -3), ("1e-3", 0.001), (".5", 0.5), ("1,200.50", 1200.5),
    ("5/36", 5 / 36), ("$12", 12), ("12 EUR", 12), ("-€3", -3), ("12.5%", 12.5), (7, 7.0),
])
def test_coerce_number(raw, val):
    assert dsh.coerce_number(raw) == pytest.approx(val)


@pytest.mark.parametrize("raw", ["about 12", "12 kg", "12,5", True, float("nan"), "1/0", None])
def test_coerce_number_rejects(raw):
    with pytest.raises(ValueError):
        dsh.coerce_number(raw)


def _scorer():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "scoring.py")
    if not os.path.exists(path):
        pytest.skip("no scoring.py next to test_dsh.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("scoring_ref", path)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc


def _outcome(fn, x):
    try:
        return fn(x)
    except ValueError:
        return "error"


def test_coerce_agrees_with_scorer():
    import numpy as np
    sc = _scorer()
    for raw in ["12.5", "-3", "1,200.50", "5/36", "$12", "12 EUR", "12.5%", "12,5", "about 12",
                "1/0", "+ 4", "1e3", "€-3", "\u22123.5", 3, 2.5, True, np.float32(1.5),
                np.int64(4), np.bool_(True), np.float64("nan"), None, [1]]:
        assert _outcome(dsh.coerce_number, raw) == _outcome(sc.coerce_number, raw), raw
    for raw in [" North ", "yes", 3, np.int64(2), 2.0, True, None]:
        ours, ref = _outcome(dsh.coerce_label, raw), _outcome(sc.coerce_label, raw)
        assert (ours == "error") == (ref == "error"), raw
        if ours != "error":
            assert ours.casefold() == ref, raw
    for raw in [[1, 2], (1, 2), np.array([1, 2]), np.zeros((2, 2)), "12", 3]:
        assert (_outcome(dsh.coerce_list, raw) == "error") == (_outcome(sc.coerce_list, raw) == "error")


def test_to_answer_is_accepted_by_scorer():
    import numpy as np
    sc = _scorer()
    tasks = {
        "number": ({"kind": "numeric", "abs_tol": 0.01, "rel_tol": 0}, 3.0, np.float64(3.0)),
        "category": ({"kind": "category"}, "CAR", "car "),
        "list": ({"kind": "list"}, ["Lyon", "Nice"], ("lyon", "NICE")),
        "vector": ({"kind": "normalized", "metric": "smape", "baseline": 20, "reference": 10,
                    "length": 2}, [1.0, 2.0], np.array([1.0, 2.0])),
        "predictions": ({"kind": "normalized", "metric": "macro_f1", "baseline": 0.4,
                         "reference": 0.9, "length": 2, "labels": ["yes", "no"]},
                        ["yes", "no"], ["Yes", "no"]),
    }
    for kind, (scoring, gold, value) in tasks.items():
        answer = dsh.to_answer(value, kind)
        s, err = sc.score_answer({"scoring": scoring, "answer": gold}, answer)
        assert err is None and s == 1.0, kind


def test_to_answer():
    assert dsh.to_answer([3], "number") == 3.0
    assert dsh.to_answer(" north ", "category") == "north"
    assert dsh.to_answer(["a", 2], "list") == ["a", "2"]
    assert dsh.to_answer([1, 2.5], "vector") == [1.0, 2.5]
    assert dsh.to_answer(["yes", 1, 2.5], "predictions") == ["yes", 1, 2.5]
    with pytest.raises(ValueError):
        dsh.to_answer("12 kg", "number")
    with pytest.raises(ValueError):
        dsh.to_answer(3.0, "vector")
    with pytest.raises(ValueError):
        dsh.to_answer([1.5], "list")


def test_csv_overview_small_files_in_full():
    dd = "column,description\n" + "".join(f"c{i},col {i}\n" for i in range(10))
    txt = dsh.csv_overview({"data_dictionary.csv": dd})
    assert "All rows" in txt and "c9,col 9" in txt


# --- parsing free text --------------------------------------------------------------------

@pytest.mark.parametrize("text,val", [
    ("Step 1: 3*4=12.\n#### 42.5", 42.5), ("so the answer is 1,200.50 dollars", 1200.5),
    ("#### **$3.20**", 3.2), ("#### 5/36", 5 / 36), ("The mean is 7. Final: 8", 8.0),
    ("#### about 12 kg", 12.0), ("#### \u22124.5", -4.5), ("no number here", None),
])
def test_parse_number(text, val):
    assert dsh.parse_number(text) == (pytest.approx(val) if val is not None else None)


def test_parse_label():
    assert dsh.parse_label("reasoning...\n#### `North`.") == "North"
    assert dsh.parse_label("#### The winner is South", choices=["north", "south"]) == "south"
    assert dsh.parse_label("#### nothing", choices=["north"]) is None


def test_parse_list():
    assert dsh.parse_list("#### [1.5, 2, 3.25]") == [1.5, 2.0, 3.25]
    assert dsh.parse_list("#### yes, no, 'yes'", "label") == ["yes", "no", "yes"]
    assert dsh.parse_list("#### 1200.5; 3;4") == [1200.5, 3.0, 4.0]
    assert dsh.parse_list("#### 1,200.5") == [1.0, 200.5]  # commas separate items


# --- budget -------------------------------------------------------------------------------

def test_budget():
    b = dsh.Budget.for_tasks([{"time_budget_s": 5}, {"time_budget_s": 5}], safety=0.5)
    assert b.total == 5 and b.has(4) and not b.has(6) and b.elapsed() < 1


# --- load_llm (torch and transformers replaced by stubs: no model, no download) -----------

def _stub_loaders(monkeypatch):
    import types
    calls = []

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    class _Auto:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if "dtype" not in kw:  # the tokenizer
                return types.SimpleNamespace(pad_token=None, eos_token="</s>", chat_template="")
            return _Model()

    torch = types.SimpleNamespace(float16="fp16", float32="fp32")
    transformers = types.SimpleNamespace(AutoModelForCausalLM=_Auto, AutoTokenizer=_Auto)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(dsh, "cached_revision", lambda model_id: "abc123")
    return calls


@pytest.mark.parametrize("model_id,remote", [("Qwen/Qwen3-1.7B", False),
                                             ("internlm/internlm2_5-1_8b-chat", True)])
def test_load_llm_pins_revision_and_trusts_remote_code_when_needed(monkeypatch, model_id, remote):
    calls = _stub_loaders(monkeypatch)
    dsh.load_llm(model_id, device="cpu")
    assert len(calls) == 2 and all(kw["revision"] == "abc123" for kw in calls)
    assert all(kw.get("trust_remote_code", False) is remote for kw in calls)


def test_load_llm_trust_remote_code_override(monkeypatch):
    calls = _stub_loaders(monkeypatch)
    dsh.load_llm("Qwen/Qwen3-1.7B", device="cpu", trust_remote_code=True)
    assert all(kw["trust_remote_code"] is True for kw in calls)

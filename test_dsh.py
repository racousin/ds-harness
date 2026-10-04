"""Tests of the kit without a model:  python -m pytest -q test_dsh.py"""
import os

import pytest

import dsh
import scoring


@pytest.mark.parametrize("x", [3, -2.5, "41.07", " -3 ", "1e-3", "+.5", 0])
def test_to_number_agrees_with_the_scorer(x):
    assert dsh.to_number(x) == scoring.coerce_answer(x)


@pytest.mark.parametrize("x", [True, "1,200", "12 kg", "about 3", "nan", float("inf"), None, [1], "5/36"])
def test_to_number_refuses_what_the_scorer_refuses(x):
    with pytest.raises(ValueError):
        dsh.to_number(x)
    with pytest.raises(ValueError):
        scoring.coerce_answer(x)


def test_tagged_and_last_number():
    r = "Step 1: 12 * 3 = 36\nStep 2: 36 + 5.5 = 41.5\nANSWER: 41.5"
    assert dsh.tagged_number(r) == 41.5
    assert dsh.tagged_number("**ANSWER: 1,200.25 euros**") == 1200.25
    assert dsh.tagged_number("no tag here 7") is None
    assert dsh.last_number("so the total is 1,234.5 and that is it") == 1234.5
    assert dsh.last_number("−3 degrees") == -3
    assert dsh.last_number("nothing") is None


def test_calculator():
    assert dsh.calculator("15.18 - 15.9") == pytest.approx(-0.72)
    assert dsh.calculator("sqrt(2) * 2**10") == pytest.approx(2 ** 0.5 * 1024)
    assert dsh.calculator("comb(10, 3)") == 120
    assert dsh.calculator("48213 × 7319") == 48213 * 7319
    for bad in ("__import__('os')", "2 ^ 3", "x + 1", "open('f')"):
        with pytest.raises(ValueError):
            dsh.calculator(bad)


def test_parse_tool_call():
    assert dsh.parse_tool_call('I compute <call tool="calculator">2 + 2</call>') == ("calculator", "2 + 2")
    assert dsh.parse_tool_call("<call tool=calculator> 3*4 </call>") == ("calculator", "3*4")
    assert dsh.parse_tool_call("ANSWER: 4") is None


def test_run_python_sees_the_files(tmp_path):
    p = tmp_path / "loans.csv"
    p.write_text("id,amt\n1,10\n2,32\n")
    r = dsh.run_python("import pandas as pd\nRESULT = pd.read_csv('loans.csv')['amt'].mean()", [str(p)])
    assert r.ok and r.value == 21
    r = dsh.run_python("x = 1", [])
    assert not r.ok and "RESULT" in r.error
    r = dsh.run_python("while True: pass", [], timeout_s=1)
    assert not r.ok and "Timeout" in r.error


def test_file_preview(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("".join(f"{i},{i * i}\n" for i in range(20)))
    v = dsh.file_preview([str(p)], max_lines=3)
    assert "a.csv (20 lines)" in v and "2,4" in v and "5,25" not in v


def test_clock():
    c = dsh.Clock(40, margin=6)
    assert 33 < c.left() <= 34


def test_stage1_parser_on_a_fake_model(monkeypatch, tmp_path):
    import stage1_direct

    class Fake:
        def chat(self, convs, **kw):
            return ["12 + 30 = 42\nANSWER: 42", "the result is 7", "no idea"]

    monkeypatch.setattr(stage1_direct.dsh, "load_llm", lambda m: Fake())
    a = stage1_direct.Agent()
    tasks = [{"id": f"t{i}", "objective": "q", "files": []} for i in range(3)]
    assert [r["answer"] for r in a.solve(tasks)] == [42, 7, 0]


def test_dev_set_is_well_formed():
    import json
    path = os.path.join(os.path.dirname(__file__), "dev.json")
    if not os.path.exists(path):
        pytest.skip("dev.json not downloaded")
    tasks = json.load(open(path))
    for t in tasks:
        scoring.validate_task(t)
    assert len({t["id"] for t in tasks}) == len(tasks)

"""Check the benchmark itself: gold answers run, and the scorer separates right from wrong."""

import pytest

from analyst_agent.agent import AgentResult
from eval.questions import QUESTIONS
from eval.run_eval import evaluate, to_markdown
from eval.scoring import gold_answer, score, values_match
from fakes import ScriptedProvider, final

DB = "data/chinook.sqlite"


def test_ids_are_unique():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("question", [q for q in QUESTIONS if q.gold_sql], ids=lambda q: q.id)
def test_gold_answers_compute(question):
    assert gold_answer(question, DB) is not None


@pytest.mark.parametrize("actual,expected,ok", [
    (59, 59, True),
    ("59", 59, True),
    ("$2,328.60", 2328.6, True),
    (2328.61, 2328.6, True),
    (3000, 2328.6, False),
    ("rock", "Rock", True),
    ("Rock ", "Rock", True),
    ("Metal", "Rock", False),
    ("not a number", 5, False),
    (True, 1, False),
    (0.0, -0.0056, False),
    (0.03, 0.0344, False),
    (0.0345, 0.0344, True),
])
def test_values_match(actual, expected, ok):
    assert values_match(actual, expected) is ok


@pytest.mark.parametrize("question", [
    q for q in QUESTIONS
    if q.kind == "scalar" and q.gold_sql and "LIMIT 1" in q.gold_sql and "ORDER BY" in q.gold_sql
], ids=lambda q: q.id)
def test_top_one_questions_have_a_unique_winner(question):
    """A tie would make a correct answer score as wrong."""
    import re
    import sqlite3

    match = re.search(r"ORDER BY (.+?)(?: DESC)? LIMIT 1", question.gold_sql)
    sql = question.gold_sql.replace(match.group(0), f"ORDER BY {match.group(1)} DESC LIMIT 2")
    head = sql.split(" FROM ", 1)[0]
    sql = sql.replace(head, f"{head}, {match.group(1)}", 1)
    rows = sqlite3.connect(DB).execute(sql).fetchall()
    # Two rows may tie if they give the same answer, like the two playlists named Music.
    assert rows[0][0] == rows[1][0] or rows[0][-1] != rows[1][-1], rows


def _answered(value, answerable=True):
    return AgentResult(question="q", status="answered", value=value, answerable=answerable)


def test_harness_gives_full_marks_to_a_correct_plan_and_zero_to_a_wrong_one():
    questions = [q for q in QUESTIONS if q.kind in {"scalar", "list"}][:8]

    def script(right: bool):
        turns = []
        for q in questions:
            gold = gold_answer(q, DB)
            wrong = [str(x) + "_wrong" for x in gold] if isinstance(gold, list) else -12345
            turns.append(final(gold if right else wrong))
        return ScriptedProvider(turns)

    good = evaluate(script(True), DB, questions, verbose=False)
    bad = evaluate(script(False), DB, questions, verbose=False)
    assert good["overall"]["accuracy"] == 1.0
    assert bad["overall"]["accuracy"] == 0.0
    assert "Overall" in to_markdown(good)


def test_refusal_scoring_goes_both_ways():
    refuse = next(q for q in QUESTIONS if q.kind == "refusal")
    answerable = next(q for q in QUESTIONS if q.kind == "scalar")
    assert score(refuse, _answered(None, answerable=False), None)[0] is True
    assert score(refuse, _answered(100), None)[0] is False
    assert score(answerable, _answered(None, answerable=False), 59)[0] is False


def test_unfinished_run_scores_wrong():
    question = next(q for q in QUESTIONS if q.kind == "scalar")
    assert score(question, AgentResult(question="q", status="max_steps", value=59), 59)[0] is False

"""Compare an agent's machine-readable value with the gold answer."""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

from analyst_agent.agent import AgentResult

from .questions import Question

REL_TOLERANCE = 0.01
# Allows cent-level rounding on money, but must not apply to small values such as a
# correlation or a slope, where 0.01 would accept answers that are far off.
ABS_TOLERANCE = 0.01
ABS_TOLERANCE_MIN_MAGNITUDE = 10.0


def gold_answer(question: Question, database: str | Path):
    if question.kind == "refusal":
        return None
    conn = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(question.gold_sql, conn)
    finally:
        conn.close()
    if question.transform is not None:
        return question.transform(frame)
    first = frame.iloc[:, 0].tolist()
    return first if question.kind in {"list", "set"} else first[0]


def _as_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[,$%\s]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def values_match(actual, expected) -> bool:
    expected_number = _as_number(expected)
    if expected_number is not None and not isinstance(expected, str):
        actual_number = _as_number(actual)
        if actual_number is None:
            return False
        abs_tol = ABS_TOLERANCE if abs(expected_number) >= ABS_TOLERANCE_MIN_MAGNITUDE else 0.0
        return math.isclose(actual_number, expected_number, rel_tol=REL_TOLERANCE, abs_tol=abs_tol)
    return _normalize_text(actual) == _normalize_text(expected)


def score(question: Question, result: AgentResult, expected) -> tuple[bool, str]:
    """Return (correct, reason). Refusing an answerable question is an error, and so is answering a refusal."""
    if result.status != "answered":
        return False, f"did not finish: {result.status}"
    if question.kind == "refusal":
        return (result.answerable is False, "refused" if result.answerable is False else "should have refused")
    if result.answerable is False:
        return False, "refused an answerable question"

    actual = result.value
    if question.kind == "scalar":
        if isinstance(actual, list) and len(actual) == 1:
            actual = actual[0]
        ok = values_match(actual, expected)
    else:
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False, f"expected {len(expected)} items, got {actual!r}"
        if question.kind == "list":
            ok = all(values_match(a, e) for a, e in zip(actual, expected))
        else:
            remaining = list(expected)
            ok = True
            for item in actual:
                match = next((e for e in remaining if values_match(item, e)), None)
                if match is None:
                    ok = False
                    break
                remaining.remove(match)
    return ok, "match" if ok else f"expected {expected!r}, got {actual!r}"

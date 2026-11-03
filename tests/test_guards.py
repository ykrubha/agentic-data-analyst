import os

import pandas as pd
import pytest

from analyst_agent.sandbox import CodeRejected, check_code, run_python
from analyst_agent.sql_guard import ReadOnlyDatabase, SqlRejected, validate_sql
from eval.guard_benchmark import ATTACKS, run as guard_run

DB = "data/chinook.sqlite"


@pytest.mark.parametrize("attack", ATTACKS)
def test_attacks_are_rejected(attack):
    db = ReadOnlyDatabase(DB)
    with pytest.raises(Exception):
        db.query(attack)


def test_guard_benchmark_is_clean():
    report = guard_run(DB)
    assert report["escaped"] == []
    assert report["engine_only_escaped"] == []
    assert report["false_rejections"] == []
    assert report["database_file_unchanged"] is True


@pytest.mark.parametrize("query", [
    "SELECT 1",
    "SELECT Name FROM Genre LIMIT 3",
    "WITH a AS (SELECT 1 x) SELECT x FROM a",
    "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c WHERE n < 3) SELECT n FROM c",
    "SELECT 1 UNION SELECT 2",
    "SELECT Name, ROW_NUMBER() OVER (ORDER BY Name) FROM Genre",
])
def test_legitimate_reads_pass(query):
    assert validate_sql(query) == query
    ReadOnlyDatabase(DB).query(query)


def test_row_limit_truncates():
    result = ReadOnlyDatabase(DB, max_rows=10).query("SELECT * FROM Track")
    assert len(result.frame) == 10 and result.truncated


def test_runaway_query_is_stopped():
    db = ReadOnlyDatabase(DB, timeout_s=0.3)
    forever = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c) SELECT MAX(n) FROM c"
    with pytest.raises(Exception, match="limit"):
        db.query(forever)


@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from socket import socket",
    "import sys",
    "open('/etc/passwd')",
    "eval('1')",
    "__import__('os')",
    "().__class__.__bases__",
    "pd.read_csv('/etc/passwd')",
    "df.to_csv('/tmp/out')",
    "np.fromfile('/etc/passwd')",
    "getattr(pd, 'read_csv')",
])
def test_unsafe_code_is_rejected(code):
    with pytest.raises(CodeRejected):
        check_code(code)


def test_sandbox_returns_stdout_and_uses_frames():
    frames = {"df": pd.DataFrame({"x": [1, 2, 3]})}
    result = run_python("print(df['x'].sum())", frames=frames)
    assert result.ok and result.stdout.strip() == "6"


def test_sandbox_reports_exceptions():
    result = run_python("1 / 0")
    assert not result.ok and "ZeroDivisionError" in result.stderr


def test_sandbox_kills_infinite_loop():
    result = run_python("while True:\n    pass", timeout_s=2, cpu_s=1)
    assert not result.ok


def test_sandbox_enforces_memory_limit():
    result = run_python("x = bytearray(4 * 1024 ** 3)\nprint('allocated')", memory_mb=800)
    assert not result.ok and "allocated" not in result.stdout


def test_sandbox_hides_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value")
    # os cannot be imported, so reach the environment through a sanctioned path if one exists.
    result = run_python("import pandas\nprint(pandas.io.common.os.environ.get('ANTHROPIC_API_KEY'))")
    assert "secret-value" not in result.stdout
    assert os.environ["ANTHROPIC_API_KEY"] == "secret-value"

"""Read-only SQL enforcement.

Two independent layers, so a parser bug alone cannot cause a write:

1. ``validate_sql`` parses the statement with sqlglot and allows exactly one
   SELECT (or WITH ... SELECT) that references no denied function or system table.
2. ``ReadOnlyDatabase`` opens SQLite with ``mode=ro`` and installs an authorizer
   that rejects everything other than reads, so even a query that slipped past
   the parser is refused by the database engine.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

# sqlglot logs a warning when it falls back to a Command node. We reject those nodes anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

DENIED_FUNCTIONS = {
    "load_extension",
    "readfile",
    "writefile",
    "edit",
    "fts3_tokenizer",
    "randomblob",  # can allocate arbitrary memory
    "zeroblob",
}
SYSTEM_TABLE_PREFIXES = ("sqlite_", "pragma_")


class SqlRejected(ValueError):
    """The statement is not an allowed read-only query."""


class SqlExecutionError(RuntimeError):
    """The statement was allowed but failed or exceeded a limit when it ran."""


def validate_sql(query: str) -> str:
    """Return the query if it is one safe SELECT; raise ``SqlRejected`` otherwise."""
    if not query or not query.strip():
        raise SqlRejected("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(query, read="sqlite") if s is not None]
    except (ParseError, TokenError) as err:
        raise SqlRejected(f"Could not parse SQL: {str(err).splitlines()[0]}") from err

    if len(statements) != 1:
        raise SqlRejected(f"Expected exactly one statement, got {len(statements)}.")
    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        raise SqlRejected(f"Only SELECT is allowed, got {type(tree).__name__}.")

    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
        exp.Attach, exp.Detach, exp.Pragma, exp.Command, exp.Merge, exp.TruncateTable,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise SqlRejected(f"{type(node).__name__} is not allowed.")
        if isinstance(node, (exp.Anonymous, exp.Func)):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in DENIED_FUNCTIONS:
                raise SqlRejected(f"Function {name}() is not allowed.")
        if isinstance(node, exp.Table) and node.name.lower().startswith(SYSTEM_TABLE_PREFIXES):
            raise SqlRejected(f"System table {node.name} is not allowed.")
    return query


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)


def _authorizer(action, arg1, arg2, _db, _trigger):
    if action in _ALLOWED_ACTIONS or action == _RECURSIVE:
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and (arg1 or "").lower().startswith("sqlite_"):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    frame: pd.DataFrame
    truncated: bool
    elapsed_s: float


class ReadOnlyDatabase:
    """A SQLite database that can only be read, with time and row limits."""

    def __init__(self, path: str | Path, max_rows: int = 5000, timeout_s: float = 10.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Database not found: {self.path}")
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        conn.set_authorizer(_authorizer)
        return conn

    def query(self, sql: str) -> QueryResult:
        validate_sql(sql)
        conn = self._connect()
        started = time.monotonic()
        deadline = started + self.timeout_s
        # Returning non-zero from the progress handler aborts a runaway query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cursor = conn.execute(sql)
            columns = [c[0] for c in cursor.description or []]
            rows = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as err:
            if time.monotonic() > deadline:
                raise SqlExecutionError(f"Query exceeded {self.timeout_s:.0f}s limit.") from err
            raise SqlExecutionError(str(err)) from err
        finally:
            conn.close()
        truncated = len(rows) > self.max_rows
        frame = pd.DataFrame(rows[: self.max_rows], columns=columns)
        return QueryResult(frame, truncated, time.monotonic() - started)

    def tables(self) -> list[str]:
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def describe(self, table: str) -> dict:
        if table not in self.tables():
            raise SqlExecutionError(f"Unknown table {table!r}. Available: {', '.join(self.tables())}")
        conn = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        try:
            columns = [
                {"name": r[1], "type": r[2], "primary_key": bool(r[5])}
                for r in conn.execute(f'PRAGMA table_info("{table}")')
            ]
            foreign_keys = [
                {"column": r[3], "references": f"{r[2]}.{r[4]}"}
                for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')
            ]
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            cursor = conn.execute(f'SELECT * FROM "{table}" LIMIT 3')
            names = [c[0] for c in cursor.description]
            sample = [dict(zip(names, row)) for row in cursor.fetchall()]
        finally:
            conn.close()
        return {"table": table, "row_count": count, "columns": columns,
                "foreign_keys": foreign_keys, "sample_rows": sample}

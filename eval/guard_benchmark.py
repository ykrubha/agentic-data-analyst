"""Measure the SQL guard without any LLM: attacks must fail, gold queries must pass.

    python -m eval.guard_benchmark

Each attack also runs against a throwaway copy of the database, and the file's
hash is compared afterwards, so the check covers real effects and not only an
exception being raised.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from analyst_agent.sql_guard import ReadOnlyDatabase, SqlExecutionError, SqlRejected, validate_sql

from .questions import QUESTIONS

ATTACKS = [
    "DROP TABLE Invoice",
    "DELETE FROM Invoice",
    "UPDATE Track SET UnitPrice = 0",
    "INSERT INTO Genre (GenreId, Name) VALUES (999, 'x')",
    "CREATE TABLE stolen AS SELECT * FROM Customer",
    "ALTER TABLE Customer ADD COLUMN pwned TEXT",
    "CREATE INDEX i ON Track (Name)",
    "CREATE VIEW v AS SELECT 1",
    "CREATE TRIGGER t AFTER INSERT ON Genre BEGIN DELETE FROM Invoice; END",
    "REPLACE INTO Genre (GenreId, Name) VALUES (1, 'x')",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "BEGIN; DELETE FROM Invoice; COMMIT",
    "SELECT 1; DROP TABLE Invoice",
    "SELECT 1; SELECT 2",
    "SELECT * FROM Customer; -- harmless\nDELETE FROM Customer",
    "/* comment */ DELETE FROM Invoice",
    "WITH x AS (SELECT 1) DELETE FROM Invoice",
    "WITH x AS (SELECT 1) UPDATE Track SET UnitPrice = 0",
    "PRAGMA writable_schema = 1",
    "PRAGMA table_info(Customer)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "DETACH DATABASE main",
    "SELECT load_extension('/tmp/evil.so')",
    "SELECT LOAD_EXTENSION('/tmp/evil.so')",
    "SELECT readfile('/etc/passwd')",
    "SELECT writefile('/tmp/x', 'data')",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM SQLITE_SCHEMA",
    "SELECT sql FROM sqlite_temp_master",
    "SELECT * FROM pragma_table_info('Customer')",
    "SELECT randomblob(2000000000)",
    "SELECT zeroblob(2000000000)",
    "",
    "   ",
    "not sql at all",
]
# Harmless to the data, so only the parser's one-statement policy rejects it.
PARSER_POLICY_ONLY = {"SELECT 1; SELECT 2"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(database: str | Path = "data/chinook.sqlite") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "copy.sqlite"
        shutil.copy(database, copy)
        before = _sha(copy)
        db = ReadOnlyDatabase(copy)

        blocked_at_parser = 0
        blocked_at_engine = []
        escaped = []
        for attack in ATTACKS:
            try:
                validate_sql(attack)
            except SqlRejected:
                blocked_at_parser += 1
                continue
            try:
                db.query(attack)
                escaped.append(attack)
            except (SqlRejected, SqlExecutionError):
                blocked_at_engine.append(attack)

        # Prove the second layer works alone by bypassing the parser.
        engine_only_blocked = 0
        engine_only_escaped = []
        for attack in ATTACKS:
            if not attack.strip() or attack in PARSER_POLICY_ONLY:
                continue
            conn = db._connect()
            try:
                conn.executescript(attack)
                engine_only_escaped.append(attack)
            except Exception:
                engine_only_blocked += 1
            finally:
                conn.close()

        false_rejections = []
        gold = [q for q in QUESTIONS if q.gold_sql]
        for question in gold:
            try:
                db.query(question.gold_sql)
            except (SqlRejected, SqlExecutionError) as err:
                false_rejections.append({"id": question.id, "error": str(err)})

        file_unchanged = before == _sha(copy)

    return {
        "attacks": len(ATTACKS),
        "blocked_at_parser": blocked_at_parser,
        "blocked_at_engine": blocked_at_engine,
        "escaped": escaped,
        "engine_only_attempted": engine_only_blocked + len(engine_only_escaped),
        "engine_only_blocked": engine_only_blocked,
        "engine_only_escaped": engine_only_escaped,
        "gold_queries": len(gold),
        "false_rejections": false_rejections,
        "database_file_unchanged": file_unchanged,
    }


def main() -> int:
    report = run()
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "guard_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if report["escaped"] or report["false_rejections"] or not report["database_file_unchanged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

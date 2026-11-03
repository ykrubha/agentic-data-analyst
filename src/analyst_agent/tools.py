"""The tools the model can call, with JSON Schemas and an executor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .sandbox import CodeRejected, run_python
from .sql_guard import ReadOnlyDatabase, SqlExecutionError, SqlRejected

TOOL_SCHEMAS = [
    {
        "name": "list_tables",
        "description": "List every table in the database. Call this first if you have not seen the schema.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "describe_table",
        "description": "Get a table's columns, foreign keys, row count and three sample rows.",
        "parameters": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "Exact table name."}},
            "required": ["table"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Run one read-only SQLite SELECT. Returns up to 20 preview rows. The full result is "
            "kept as a pandas DataFrame named by save_as, which run_python can use."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SELECT or WITH ... SELECT."},
                "save_as": {
                    "type": "string",
                    "description": "Python identifier for the resulting DataFrame, such as monthly_sales.",
                },
            },
            "required": ["query", "save_as"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run Python for analysis SQL cannot do cleanly: statistics, regression, reshaping, charts. "
            "Saved DataFrames are preloaded as globals; pd, np and plt are already imported. "
            "print() anything you need to see. Save charts with plt.savefig('name.png'). "
            "No file reads or network access."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Finish the task. Call exactly once, after you have verified the result with data.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Plain-language answer with the key numbers."},
                "value": {
                    "description": (
                        "The machine-readable result: a number, a string, or a list of them, in "
                        "the order the question asks for. Use null if the question cannot be "
                        "answered from this data."
                    ),
                },
                "answerable": {
                    "type": "boolean",
                    "description": "False if the database does not contain what the question needs.",
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Interpretation choices the reader should know about.",
                },
            },
            "required": ["answer", "value", "answerable"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class ToolExecutor:
    """Runs tool calls for one agent session and holds its saved DataFrames."""

    db: ReadOnlyDatabase
    output_dir: Path | None = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    charts: list[str] = field(default_factory=list)
    sql_history: list[str] = field(default_factory=list)
    preview_rows: int = 20

    def execute(self, name: str, arguments: dict) -> ToolResult:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(f"Unknown tool {name!r}.", is_error=True)
        try:
            return handler(**arguments)
        except TypeError as err:
            return ToolResult(f"Bad arguments for {name}: {err}", is_error=True)
        except (SqlRejected, SqlExecutionError, CodeRejected) as err:
            return ToolResult(f"{type(err).__name__}: {err}", is_error=True)

    def _tool_list_tables(self) -> ToolResult:
        return ToolResult(json.dumps({"tables": self.db.tables()}))

    def _tool_describe_table(self, table: str) -> ToolResult:
        return ToolResult(json.dumps(self.db.describe(table), default=str))

    def _tool_run_sql(self, query: str, save_as: str = "result") -> ToolResult:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,40}", save_as):
            return ToolResult("save_as must be a valid Python identifier.", is_error=True)
        result = self.db.query(query)
        self.frames[save_as] = result.frame
        self.sql_history.append(query)
        preview = result.frame.head(self.preview_rows)
        return ToolResult(json.dumps({
            "saved_as": save_as,
            "row_count": len(result.frame),
            "columns": list(result.frame.columns),
            "truncated_at_row_limit": result.truncated,
            "preview": json.loads(preview.to_json(orient="records", date_format="iso")),
        }))

    def _tool_run_python(self, code: str) -> ToolResult:
        result = run_python(code, frames=self.frames, output_dir=self.output_dir)
        self.charts.extend(result.charts)
        payload = {"stdout": result.stdout, "charts": result.charts}
        if not result.ok:
            payload["stderr"] = result.stderr
        return ToolResult(json.dumps(payload), is_error=not result.ok)

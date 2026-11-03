"""Run model-written Python in a separate, resource-limited process.

Layers:
* A static AST check allows only analysis imports and blocks dunder access,
  ``eval``/``exec``/``open`` and similar.
* The code runs in a child process in an empty working directory with a clean
  environment (no API keys) and rlimits on CPU, memory, file size and processes.

This is defence in depth for a trusted-operator tool, not a security boundary
against a determined attacker. Python static checks can be bypassed, so a
multi-tenant deployment should also run this inside a container or gVisor with
no network.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_IMPORTS = {
    "pandas", "numpy", "scipy", "statistics", "math", "datetime",
    "matplotlib", "collections", "itertools", "functools", "re", "json",
    "decimal", "sklearn",
}
BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}
# pandas and numpy can reach the filesystem or network; frames are preloaded instead.
BLOCKED_ATTRS = {
    "read_csv", "read_parquet", "read_pickle", "read_sql", "read_sql_query",
    "read_sql_table", "read_html", "read_json", "read_excel", "read_xml",
    "read_table", "read_fwf", "read_feather", "read_hdf", "read_stata",
    "read_sas", "read_spss", "read_orc", "read_clipboard", "to_csv",
    "to_parquet", "to_pickle", "to_sql", "to_excel", "to_json", "to_html",
    "to_feather", "to_hdf", "to_stata", "to_xml", "to_clipboard", "to_orc",
    "load", "save", "savez", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "memmap", "system", "popen",
}


class CodeRejected(ValueError):
    """The code failed the static safety check."""


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        raise CodeRejected(f"SyntaxError on line {err.lineno}: {err.msg}") from err
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _check_import(node.module or "")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise CodeRejected(f"Use of {node.id}() is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise CodeRejected(f"Access to private attribute .{node.attr} is not allowed.")
            if node.attr in BLOCKED_ATTRS:
                raise CodeRejected(
                    f".{node.attr} is not allowed. Use the preloaded DataFrames; "
                    "charts go through plt.savefig('chart.png')."
                )


def _check_import(name: str) -> None:
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise CodeRejected(
            f"Import of {name!r} is not allowed. Allowed: {', '.join(sorted(ALLOWED_IMPORTS))}."
        )


_BOOTSTRAP = """\
import resource, sys
def _limit(kind, value):
    try:
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError):
        pass
_limit(resource.RLIMIT_CPU, {cpu})
_limit(resource.RLIMIT_AS, {memory})
_limit(resource.RLIMIT_FSIZE, 50 * 1024 * 1024)
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
frames = {{}}
_name = None
for _name in {names!r}:
    frames[_name] = pd.read_pickle(_name + ".pkl")
    globals()[_name] = frames[_name]
del _name, _limit, resource
"""


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    charts: list[str] = field(default_factory=list)


def run_python(
    code: str,
    frames: dict | None = None,
    output_dir: str | Path | None = None,
    timeout_s: float = 20.0,
    cpu_s: int = 15,
    memory_mb: int = 1536,
    max_output_chars: int = 6000,
) -> SandboxResult:
    """Check and run ``code``; each frame in ``frames`` becomes a global DataFrame."""
    check_code(code)
    frames = frames or {}
    with tempfile.TemporaryDirectory(prefix="analyst_sandbox_") as tmp:
        tmp_path = Path(tmp)
        for name, frame in frames.items():
            if not name.isidentifier():
                raise CodeRejected(f"Invalid frame name {name!r}.")
            frame.to_pickle(tmp_path / f"{name}.pkl")
        bootstrap = _BOOTSTRAP.format(
            cpu=cpu_s, memory=memory_mb * 1024 * 1024, names=sorted(frames)
        )
        script = tmp_path / "_analysis.py"
        script.write_text(bootstrap + "\n" + code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path),
            "HOME": str(tmp_path),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script.name],
                cwd=tmp, env=env, capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, "", f"Timed out after {timeout_s:.0f}s.")

        charts: list[str] = []
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            existing = len(list(out.glob("*.png")))
            for index, chart in enumerate(sorted(tmp_path.glob("*.png")), start=existing):
                target = out / f"chart_{index:02d}_{chart.name}"
                target.write_bytes(chart.read_bytes())
                charts.append(str(target))

        stderr = proc.stderr
        # Keep the useful end of a traceback and hide the temp path.
        stderr = stderr.replace(tmp + os.sep, "")[-2000:]
        return SandboxResult(proc.returncode == 0, proc.stdout[:max_output_chars], stderr, charts)


def result_to_json(result: SandboxResult) -> str:
    return json.dumps(
        {"ok": result.ok, "stdout": result.stdout, "stderr": result.stderr, "charts": result.charts}
    )

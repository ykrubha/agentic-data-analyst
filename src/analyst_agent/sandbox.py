"""Run model-written Python in a separate, resource-limited process.

Layers:
* A static AST check allows only analysis imports and blocks dunder access,
  ``eval``/``exec``/``open`` and similar.
* The code runs in a child process in an empty working directory with a clean
  environment (no API keys) and rlimits on CPU, memory and file size. The memory
  limit relies on RLIMIT_AS, which only Linux enforces. On macOS it has no effect.

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


_SKIP_FONT_PROFILER = """\
import subprocess as _subprocess
_check_output = _subprocess.check_output
def _skip_system_profiler(command, *args, **kwargs):
    # On macOS matplotlib runs system_profiler to list fonts. It is slow and can return
    # output matplotlib can't parse. An OSError makes matplotlib use its font folders instead.
    if command and command[0] == "system_profiler":
        raise OSError("system_profiler skipped")
    return _check_output(command, *args, **kwargs)
_subprocess.check_output = _skip_system_profiler
"""

_PLOTTING_IMPORTS = _SKIP_FONT_PROFILER + """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""


def uses_plotting(code: str) -> bool:
    """Matplotlib is slow to import, so only load it for code that needs it."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id == "plt":
            return True
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "matplotlib" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "matplotlib":
            return True
    return False


def matplotlib_cache_dir() -> Path:
    """A font cache shared between runs, so fonts are only scanned once per machine."""
    path = Path(tempfile.gettempdir()) / f"analyst_agent_matplotlib_{os.getuid()}"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def _prepare_font_cache(cache_dir: Path) -> None:
    """Build the font cache once, outside the sandbox's CPU limit and timeout."""
    if list(cache_dir.glob("fontlist-*.json")):
        return
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(cache_dir),
           "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cache_dir)}
    script = _SKIP_FONT_PROFILER + "import matplotlib.font_manager\n"
    try:
        subprocess.run([sys.executable, "-I", "-c", script], env=env, cwd=cache_dir,
                       capture_output=True, timeout=180, check=False)
    except subprocess.TimeoutExpired:
        pass  # The sandboxed run will report its own timeout.


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
{plotting}frames = {{}}
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
        plotting = uses_plotting(code)
        mpl_dir = matplotlib_cache_dir() if plotting else tmp_path
        if plotting:
            _prepare_font_cache(mpl_dir)
        bootstrap = _BOOTSTRAP.format(
            cpu=cpu_s, memory=memory_mb * 1024 * 1024, names=sorted(frames),
            plotting=_PLOTTING_IMPORTS if plotting else "",
        )
        script = tmp_path / "_analysis.py"
        script.write_text(bootstrap + "\n" + code, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(mpl_dir),
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

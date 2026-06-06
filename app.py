"""
TechNova RAG Assistant — entry point.

This file exists so that Railpack/Nixpacks can detect this as a Python project.
The actual application source lives in 'app..py' (original uploaded filename).
The start command in railway.toml runs: streamlit run 'app..py' directly.
"""
import pathlib, sys

_original = pathlib.Path(__file__).parent / "app..py"
if not _original.exists():
    raise FileNotFoundError(f"Original app source not found: {_original}")

with open(_original, "r", encoding="utf-8") as _fh:
    _code = _fh.read()

exec(compile(_code, str(_original), "exec"))

"""Exercise Windows installer discovery and failure handling without downloads."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell installer")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/setup.ps1"


def setup_fixture(tmp_path):
    script = tmp_path / "scripts/setup.ps1"
    script.parent.mkdir()
    shutil.copy2(SCRIPT, script)
    (tmp_path / ".env.example").write_text("LLM_API_KEY=\n", encoding="utf-8")
    return script


def run_setup(script, *arguments, env=None):
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                           *arguments], capture_output=True, text=True, errors="replace", env=env, timeout=60)


def test_check_only_accepts_explicit_python_without_creating_environment(tmp_path):
    script = setup_fixture(tmp_path)
    result = run_setup(script, "-PythonExecutable", sys.executable, "-CheckOnly")
    assert result.returncode == 0, result.stderr
    assert "Python 3.10 found" in result.stdout
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / ".env").exists()


def test_missing_explicit_python_exits_with_actionable_error(tmp_path):
    result = run_setup(setup_fixture(tmp_path), "-PythonExecutable", str(tmp_path / "missing.exe"), "-CheckOnly")
    assert result.returncode != 0
    assert "Python 3.10 was not found" in result.stderr
    assert "installation.md" in result.stderr


def test_pip_failure_preserves_environment_and_never_reports_success(tmp_path):
    script = setup_fixture(tmp_path)
    fake_pip = tmp_path / "offline-pip"
    fake_pip.mkdir()
    (fake_pip / "pip.py").write_text("raise SystemExit(17)\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(fake_pip)}
    result = run_setup(script, "-PythonExecutable", sys.executable, env=environment)
    assert result.returncode != 0
    assert "pip upgrade failed" in result.stderr
    assert "Setup complete" not in result.stdout
    assert (tmp_path / ".venv/Scripts/python.exe").exists()
    assert not (tmp_path / ".env").exists()

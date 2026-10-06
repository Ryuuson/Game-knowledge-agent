$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) { throw "Run scripts/setup.ps1 first." }
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
& $projectPython scripts/doctor.py --json
if ($LASTEXITCODE -ne 0) { throw "Diagnostics failed." }
& $projectPython -m pytest tests -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Offline regression tests failed." }
Write-Host "Offline verification completed. Retrieval quality is measured separately by scripts/evaluate_retrieval.py."

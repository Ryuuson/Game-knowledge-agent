$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw "Virtual environment is missing. Run .\scripts\setup.ps1 first."
}
# The app can demonstrate keyword retrieval without credentials or a BGE index.
& $projectPython -m streamlit run UI.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }

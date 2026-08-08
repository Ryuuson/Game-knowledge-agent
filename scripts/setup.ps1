$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    py -3.10 -m venv .venv
}

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env. Fill in LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL before starting the Agent."
}

Write-Host "Setup complete. For local BGE retrieval, run: .\.venv\Scripts\python.exe build_bge_combined_index.py"

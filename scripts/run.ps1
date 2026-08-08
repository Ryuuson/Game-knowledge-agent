$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment is missing. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path ".env")) {
    throw ".env is missing. Copy .env.example to .env and configure a chat model first."
}

$ragBackend = $env:RAG_BACKEND
if (-not $ragBackend) {
    $backendLine = Get-Content ".env" | Where-Object { $_ -match "^\s*RAG_BACKEND\s*=" } | Select-Object -First 1
    if ($backendLine) {
        $ragBackend = ($backendLine -split "=", 2)[1].Trim()
    }
}
if (-not $ragBackend) {
    $ragBackend = "bge"
}

if ($ragBackend.ToLower() -eq "bge") {
    $index = Join-Path $root "data\game_knowledge_bge_combined_index.sqlite"
    if (-not (Test-Path $index)) {
        throw "The BGE index is missing. Run .\.venv\Scripts\python.exe build_bge_combined_index.py first."
    }
}

& $python -m streamlit run UI.py

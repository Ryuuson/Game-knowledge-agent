param(
    [string]$PythonExecutable,
    [switch]$WithBge,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$candidates = @()

if (Test-Path -LiteralPath $projectPython) {
    $candidates += [pscustomobject]@{ Command = $projectPython; Arguments = @() }
} elseif ($PythonExecutable) {
    $candidates += [pscustomobject]@{ Command = $PythonExecutable; Arguments = @() }
} else {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += [pscustomobject]@{ Command = "py"; Arguments = @("-3.10") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $candidates += [pscustomobject]@{ Command = "python"; Arguments = @() }
    }
}

$selectedPython = $null
foreach ($candidate in $candidates) {
    try {
        $prefix = $candidate.Arguments
        $version = & $candidate.Command @prefix -c "import sys; print('%s.%s' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and "$version".Trim() -eq "3.10") {
            $selectedPython = $candidate
            break
        }
    } catch { }
}
if (-not $selectedPython) {
    throw "Python 3.10 was not found, or the existing .venv is incompatible. Check 'py -3.10 --version' or 'python --version'. You can pass -PythonExecutable 'C:\path\to\python.exe' when creating a new environment. See docs/installation.md."
}

Write-Host "Python 3.10 found: $($selectedPython.Command)"
if ($CheckOnly) { return }
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $projectPython)) {
    $prefix = $selectedPython.Arguments
    & $selectedPython.Command @prefix -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed. Check the Python installation and folder permissions." }
}

& $projectPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. Check network/proxy settings, then rerun setup. See docs/installation.md." }
$requirements = if ($WithBge) { "requirements-bge.txt" } else { "requirements.txt" }
& $projectPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed ($requirements). Check the pip error above. The environment is kept for retry; see docs/installation.md." }

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env. Offline keyword retrieval needs no API key; configure LLM_* only for chat."
}
Write-Host "Setup complete. Run .\scripts\run.ps1 and choose BM25 for API-free retrieval."
if ($WithBge) { Write-Host "Build the local BGE index with .\.venv\Scripts\python.exe build_bge_combined_index.py" }

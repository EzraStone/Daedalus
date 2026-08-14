[CmdletBinding()]
param(
    [int] $Cases = 10,
    [int] $Seed = 1
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repository ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment is missing; create .venv and install the project first"
}

$reportPath = Join-Path $PSScriptRoot "runtime\smoke-report.json"
& $python (Join-Path $repository "harness\compare.py") `
    --cases $Cases `
    --seed $Seed `
    --out $reportPath
if ($LASTEXITCODE -ne 0) {
    throw "fidelity smoke test failed with exit code $LASTEXITCODE"
}

$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
if ($report.checked -le 0) {
    throw "fidelity smoke test did not produce any runnable circuits"
}
if ($report.unreachable -ne 0) {
    throw "fidelity smoke test could not reach $($report.unreachable) cases"
}

Write-Host "Smoke agreement: $($report.agreed)/$($report.checked) ($($report.agreement))"

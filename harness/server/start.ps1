[CmdletBinding()]
param(
    [string] $MaximumHeap = "2G"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runtime = Join-Path $PSScriptRoot "runtime"
$java = Join-Path $repository ".tools\jdk-17.0.20+8\bin\java.exe"
if (-not (Test-Path $java)) {
    & (Join-Path $PSScriptRoot "bootstrap-java.ps1")
}
if (-not (Test-Path (Join-Path $runtime "fabric-server-launch.jar"))) {
    throw "server runtime is missing; run setup.ps1 -AcceptEula first"
}

$portLine = Get-Content (Join-Path $PSScriptRoot "versions.properties") |
    Where-Object { $_ -match '^harness_port=' } |
    Select-Object -First 1
$harnessPort = ($portLine -split '=', 2)[1]

Push-Location $runtime
try {
    & $java `
        "-Xms512M" `
        "-Xmx$MaximumHeap" `
        "-Ddaedalus.harness.port=$harnessPort" `
        -jar "fabric-server-launch.jar" nogui
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

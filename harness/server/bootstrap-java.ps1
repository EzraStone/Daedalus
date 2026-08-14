[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$toolDirectory = Join-Path $repository ".tools"
New-Item -ItemType Directory -Force -Path $toolDirectory | Out-Null

function Install-PortableJdk {
    param(
        [Parameter(Mandatory)] [string] $Name,
        [Parameter(Mandatory)] [string] $Directory,
        [Parameter(Mandatory)] [string] $Url,
        [Parameter(Mandatory)] [string] $Sha256
    )

    $destination = Join-Path $toolDirectory $Directory
    if (Test-Path (Join-Path $destination "bin\java.exe")) {
        Write-Host "$Name is already installed at $destination"
        return
    }

    $archive = Join-Path $toolDirectory "$Directory.zip"
    Write-Host "Downloading $Name"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actual -ne $Sha256) {
        throw "$Name checksum mismatch: expected $Sha256, got $actual"
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $toolDirectory -Force
    if (-not (Test-Path (Join-Path $destination "bin\java.exe"))) {
        throw "$Name did not extract to $destination"
    }
}

Install-PortableJdk `
    -Name "Eclipse Temurin 17.0.20+8" `
    -Directory "jdk-17.0.20+8" `
    -Url "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20%2B8/OpenJDK17U-jdk_x64_windows_hotspot_17.0.20_8.zip" `
    -Sha256 "418497BE5CF585BDD2203D6486A565D66D3F5E992D5630D45104CB873FAB8122"

Install-PortableJdk `
    -Name "Eclipse Temurin 21.0.12+8" `
    -Directory "jdk-21.0.12+8" `
    -Url "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12%2B8/OpenJDK21U-jdk_x64_windows_hotspot_21.0.12_8.zip" `
    -Sha256 "9BA963EE2371874A74185D18BC7BB2AB9407DF7683300855ED7606E0662321D0"

Write-Host "Portable Java toolchains are ready."

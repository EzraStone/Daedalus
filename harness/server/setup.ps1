[CmdletBinding()]
param(
    [switch] $AcceptEula
)

$ErrorActionPreference = "Stop"
if (-not $AcceptEula) {
    throw "Pass -AcceptEula after reviewing https://aka.ms/MinecraftEULA"
}

$repository = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runtime = Join-Path $PSScriptRoot "runtime"
$mods = Join-Path $runtime "mods"
$versionsFile = Join-Path $PSScriptRoot "versions.properties"
$versions = @{}
foreach ($line in Get-Content -LiteralPath $versionsFile) {
    if ($line -match '^([^#=]+)=(.+)$') {
        $versions[$Matches[1]] = $Matches[2]
    }
}

$buildJava = Join-Path $repository ".tools\jdk-21.0.12+8"
if (-not (Test-Path (Join-Path $buildJava "bin\java.exe"))) {
    & (Join-Path $PSScriptRoot "bootstrap-java.ps1")
}

New-Item -ItemType Directory -Force -Path $mods | Out-Null
$oldJavaHome = $env:JAVA_HOME
$env:JAVA_HOME = $buildJava
try {
    & (Join-Path $repository "harness\mod\gradlew.bat") -p (Join-Path $repository "harness\mod") build
    if ($LASTEXITCODE -ne 0) {
        throw "Fabric mod build failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:JAVA_HOME = $oldJavaHome
}

$harnessJar = Get-ChildItem (Join-Path $repository "harness\mod\build\libs\*.jar") |
    Where-Object { $_.Name -notmatch '-sources\.jar$' } |
    Select-Object -First 1
if (-not $harnessJar) {
    throw "the built harness jar was not found"
}
Copy-Item -LiteralPath $harnessJar.FullName -Destination (Join-Path $mods "daedalus-harness.jar") -Force

$launcher = Join-Path $runtime "fabric-server-launch.jar"
if (-not (Test-Path $launcher)) {
    $launcherUrl = "https://meta.fabricmc.net/v2/versions/loader/$($versions.minecraft)/$($versions.fabric_loader)/$($versions.fabric_launcher)/server/jar"
    Invoke-WebRequest -UseBasicParsing -Uri $launcherUrl -OutFile $launcher
}

$fabricApi = Join-Path $mods "fabric-api.jar"
if (-not (Test-Path $fabricApi)) {
    $apiVersion = $versions.fabric_api
    $apiUrl = "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/$apiVersion/fabric-api-$apiVersion.jar"
    Invoke-WebRequest -UseBasicParsing -Uri $apiUrl -OutFile $fabricApi
}

Set-Content -LiteralPath (Join-Path $runtime "eula.txt") -Encoding ascii -Value "eula=true"
@'
allow-flight=true
difficulty=peaceful
enable-command-block=false
enable-query=false
enable-rcon=false
enforce-secure-profile=false
force-gamemode=true
function-permission-level=2
gamemode=creative
generate-structures=false
generator-settings={"layers":[{"block":"minecraft:air","height":1}],"biome":"minecraft:the_void"}
hardcore=false
level-name=harness-world
level-type=minecraft:flat
max-players=1
max-tick-time=-1
motd=Daedalus fidelity harness
online-mode=false
pause-when-empty-seconds=-1
player-idle-timeout=0
prevent-proxy-connections=false
pvp=false
server-ip=127.0.0.1
server-port=25565
simulation-distance=2
spawn-animals=false
spawn-monsters=false
spawn-npcs=false
spawn-protection=0
sync-chunk-writes=true
view-distance=2
'@ | Set-Content -LiteralPath (Join-Path $runtime "server.properties") -Encoding ascii

Write-Host "Pinned Fabric server is ready at $runtime"

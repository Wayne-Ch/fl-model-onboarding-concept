[CmdletBinding()]
param(
    [string]$VenvPath = "C:\fl-onboarding-venv",
    [string]$WorkspaceBase = "C:\fmo\w",
    [string]$ModelCacheDir = "C:\fmo\cache",
    [int]$Port = 8777,
    [ValidateSet("127.0.0.1", "localhost", "::1")]
    [string]$Host = "127.0.0.1",
    [ValidateSet("critical", "error", "warning", "info", "debug", "trace")]
    [string]$LogLevel = "info",
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRootResolved = (Resolve-Path -LiteralPath $RepoRoot).Path
$venvPathResolved = [System.IO.Path]::GetFullPath($VenvPath)
$workspaceBaseResolved = [System.IO.Path]::GetFullPath($WorkspaceBase)
$modelCacheResolved = [System.IO.Path]::GetFullPath($ModelCacheDir)

$venvScripts = Join-Path $venvPathResolved "Scripts"
$venvPython = Join-Path $venvScripts "python.exe"
$venvCli = Join-Path $venvScripts "fl-onboarding.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment Python not found at $venvPython. Run .\scripts\bootstrap-local-poc.ps1 first."
}

if (-not (Test-Path -LiteralPath $venvCli)) {
    throw "fl-onboarding CLI not found at $venvCli. Re-run .\scripts\bootstrap-local-poc.ps1."
}

if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}

if (-not (Test-Path -LiteralPath $workspaceBaseResolved)) {
    New-Item -ItemType Directory -Path $workspaceBaseResolved -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $modelCacheResolved)) {
    New-Item -ItemType Directory -Path $modelCacheResolved -Force | Out-Null
}

$env:PATH = "$venvScripts;$env:PATH"

Push-Location $repoRootResolved
try {
    & "fl-onboarding" "service" "serve" `
        "--host" $Host `
        "--port" "$Port" `
        "--workspace-base" $workspaceBaseResolved `
        "--model-cache-dir" $modelCacheResolved `
        "--enable-production-runner" `
        "--open-browser" `
        "--log-level" $LogLevel
    if ($LASTEXITCODE -ne 0) {
        throw "fl-onboarding service serve exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

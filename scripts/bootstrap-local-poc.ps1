[CmdletBinding()]
param(
    [string]$VenvPath = "C:\fl-onboarding-venv",
    [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
    [string]$PythonExe = "",
    [switch]$SkipWebBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-External {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$Step
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Resolve-PythonCommand {
    param([string]$Override)

    if ($Override) {
        return @{
            Command = $Override
            Prefix = @()
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{
            Command = "py"
            Prefix = @("-3.11")
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{
            Command = "python"
            Prefix = @()
        }
    }

    throw "Python was not found. Install Python 3.11+ or pass -PythonExe to this script."
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Windows x64 is required."
}

$repoRootResolved = (Resolve-Path -LiteralPath $RepoRoot).Path
$venvPathResolved = [System.IO.Path]::GetFullPath($VenvPath)
$python = Resolve-PythonCommand -Override $PythonExe

$probeCode = @'
import json
import platform
import struct
import sys

print(json.dumps({
    "major": sys.version_info[0],
    "minor": sys.version_info[1],
    "arch_bits": struct.calcsize("P") * 8,
    "machine": platform.machine(),
    "executable": sys.executable,
}))
'@

$probeOutput = & $python.Command @($python.Prefix + @("-c", $probeCode))
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run Python interpreter probe."
}
$probe = $probeOutput | ConvertFrom-Json

if ($probe.major -ne 3 -or $probe.minor -lt 11) {
    throw "Python 3.11+ is required. Detected $($probe.major).$($probe.minor)."
}

if ($probe.arch_bits -ne 64) {
    throw "Python x64 is required. Detected $($probe.arch_bits)-bit at $($probe.executable)."
}

$venvPython = Join-Path $venvPathResolved "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    $venvParent = Split-Path -Parent $venvPathResolved
    if (-not (Test-Path -LiteralPath $venvParent)) {
        New-Item -ItemType Directory -Path $venvParent -Force | Out-Null
    }
    Invoke-External -Command $python.Command -Arguments @($python.Prefix + @("-m", "venv", $venvPathResolved)) -Step "Creating virtual environment"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Virtual environment creation did not produce $venvPython."
}

Invoke-External -Command $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Step "Upgrading pip"

Push-Location $repoRootResolved
try {
    Invoke-External -Command $venvPython -Arguments @("-m", "pip", "install", "-e", ".[dev,runtime]") -Step "Installing Python dependencies"
}
finally {
    Pop-Location
}

if (-not $SkipWebBuild) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm is required to build web/. Install Node.js 20+."
    }
    Push-Location (Join-Path $repoRootResolved "web")
    try {
        Invoke-External -Command "npm" -Arguments @("ci") -Step "Installing web dependencies"
        Invoke-External -Command "npm" -Arguments @("run", "build") -Step "Building web UI"
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next command:"
Write-Host ".\scripts\run-local-ui.ps1 -VenvPath `"$venvPathResolved`""

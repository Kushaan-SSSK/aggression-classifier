# Kushaan Sharma
<#
  Create the .venv-simba and .venv-dlc environments (Python 3.10) and install ffmpeg.
  Re-runnable; existing venvs are reused.
  Usage: powershell -ExecutionPolicy Bypass -File env\setup_env.ps1 [-Cuda cu126] [-SkipDlc] [-SkipSystemInstalls]
#>
param(
    [string]$Cuda = "",
    [switch]$SkipDlc,
    [switch]$SkipSystemInstalls
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not $SkipSystemInstalls) {
    Write-Host "== Installing Python 3.10 and ffmpeg via winget (skipped if present) =="
    winget install -e --id Python.Python.3.10 --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --disable-interactivity
    # Pick up the new PATH entries in this session.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

$py = "py"
& $py -3.10 --version
if (-not $?) { throw "Python 3.10 not found via the py launcher. Install it and re-run." }

if (-not (Test-Path ".venv-simba")) { & $py -3.10 -m venv .venv-simba }
& ".venv-simba\Scripts\python.exe" -m pip install --upgrade pip wheel "setuptools<81"
& ".venv-simba\Scripts\python.exe" -m pip install -r env\requirements-simba.txt
& ".venv-simba\Scripts\python.exe" -m pip install -e .
& ".venv-simba\Scripts\python.exe" env\check_env.py

if (-not $SkipDlc) {
    if (-not (Test-Path ".venv-dlc")) { & $py -3.10 -m venv .venv-dlc }
    & ".venv-dlc\Scripts\python.exe" -m pip install --upgrade pip wheel "setuptools<81"
    if ($Cuda -ne "") {
        & ".venv-dlc\Scripts\python.exe" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    } else {
        & ".venv-dlc\Scripts\python.exe" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
    }
    & ".venv-dlc\Scripts\python.exe" -m pip install -r env\requirements-dlc.txt
    & ".venv-dlc\Scripts\python.exe" env\check_env.py
}
Write-Host "== Done. Activate with .venv-simba\Scripts\Activate.ps1 or .venv-dlc\Scripts\Activate.ps1 =="

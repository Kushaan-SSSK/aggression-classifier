# Kushaan Sharma
<#
  Run every pipeline stage in order. Pose runs in .venv-dlc, everything else in .venv-simba.
  annotate/train/infer run only once data\annotations\annotations.csv exists.
  Usage: powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-Config config\pipeline.yaml] [-SkipPose] [-Swap name1,name2] [-Only name1,name2]
#>
param(
    [string]$Config = "config\pipeline.yaml",
    [switch]$SkipPose,
    [string[]]$Swap = @(),
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$simba = ".venv-simba\Scripts\python.exe"
$dlc = ".venv-dlc\Scripts\python.exe"

function Invoke-Stage { param([string]$Exe, [string[]]$StageArgs)
    & $Exe -m behavior_pipeline @StageArgs --config $Config
    if ($LASTEXITCODE -ne 0) { throw "stage failed: $($StageArgs -join ' ')" }
}

if ($Only.Count -gt 0) { Invoke-Stage $simba @("preprocess", "--only") + $Only } else { Invoke-Stage $simba @("preprocess") }
if (-not $SkipPose) {
    if ($Only.Count -gt 0) {
        $videos = $Only | ForEach-Object { "data\processed_videos\$_.mp4" }
        Invoke-Stage $dlc (@("pose", "--videos") + $videos)
    } else { Invoke-Stage $dlc @("pose") }
}
if ($Only.Count -gt 0) { Invoke-Stage $simba (@("resident", "--videos") + $Only) } else { Invoke-Stage $simba @("resident") }
$convertArgs = @("convert")
if ($Only.Count -gt 0) { $convertArgs += @("--videos") + $Only }
if ($Swap.Count -gt 0) { $convertArgs += @("--swap") + $Swap }
Invoke-Stage $simba $convertArgs
Invoke-Stage $simba @("project")
$haveAnnotations = Test-Path "data\annotations\annotations.csv"
if ($haveAnnotations) {
    Invoke-Stage $simba @("annotate")
    Invoke-Stage $simba @("train")
    Invoke-Stage $simba @("infer")
} else {
    Write-Host "data\annotations\annotations.csv not found - skipping annotate/train/infer (fill in annotations_todo.csv first)"
}
Invoke-Stage $simba @("proximity")
Invoke-Stage $simba @("aggression-proxy")
if ($haveAnnotations) { Invoke-Stage $simba @("unsupervised") } else { Invoke-Stage $simba @("unsupervised", "--clf-slice", "Proximity") }
Invoke-Stage $simba @("annotate", "--make-todo")
Write-Host "== all stages finished =="

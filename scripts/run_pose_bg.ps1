# Kushaan Sharma
<#
  Run the pose stage for several videos as separate .venv-dlc background processes.
  Logs go to data\pose\logs\<name>.log / .err; check progress with: python -m behavior_pipeline pose --status
  Usage: powershell -ExecutionPolicy Bypass -File scripts\run_pose_bg.ps1 [-Videos name1,name2] [-MaxParallel 2] [-Mode blob]
#>
param(
    [string[]]$Videos = @(),
    [int]$MaxParallel = 1,
    [string]$Mode = "blob",
    [string]$Config = "config\pipeline.yaml"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$dlc = Join-Path $root ".venv-dlc\Scripts\python.exe"
$logDir = Join-Path $root "data\pose\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

if ($Videos.Count -eq 0) {
    $Videos = Get-ChildItem "data\processed_videos\*.mp4" | ForEach-Object { $_.BaseName }
}
$running = @()
foreach ($name in $Videos) {
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) { Start-Sleep -Seconds 10 }
    $video = Join-Path $root "data\processed_videos\$name.mp4"
    $args = "-m behavior_pipeline pose --mode $Mode --videos `"$video`" --config $Config"
    Write-Host "starting pose for $name"
    $p = Start-Process -FilePath $dlc -ArgumentList $args -WorkingDirectory $root -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $logDir "$name.log") -RedirectStandardError (Join-Path $logDir "$name.err")
    $running += $p
}
$running | ForEach-Object { $_.WaitForExit() }
Write-Host "== pose finished for: $($Videos -join ', ') =="
& $dlc -m behavior_pipeline pose --status --config $Config

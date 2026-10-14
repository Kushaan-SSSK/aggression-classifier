@echo off
rem Double-click to open the Aggression Scorer app.
cd /d "%~dp0"
if not exist ".venv-simba\Scripts\pythonw.exe" (
  echo Not installed yet. Double-click install.bat first.
  pause
  exit /b 1
)
start "" ".venv-simba\Scripts\pythonw.exe" -m behavior_pipeline gui

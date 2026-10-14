@echo off
rem Double-click to install everything on Windows (about 30 to 60 minutes).
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File env\setup_env.ps1
pause

#!/usr/bin/env bash
# Kushaan Sharma
# Install everything on macOS or Linux: Python 3.10 environments for SimBA and DeepLabCut, plus the trained model.
# Run from the program folder with: bash install.sh
set -e
cd "$(dirname "$0")/.."

PY=""
for cand in python3.10 /opt/homebrew/bin/python3.10 /usr/local/bin/python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
done

if [ "$(uname)" = "Darwin" ]; then
  if [ -z "$PY" ] || ! command -v ffmpeg >/dev/null 2>&1; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "Homebrew is needed to install Python 3.10 and ffmpeg. Install it from https://brew.sh, then run this again."
      exit 1
    fi
    brew install python@3.10 ffmpeg
    PY="$(brew --prefix)/bin/python3.10"
  fi
else
  if [ -z "$PY" ] || ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Python 3.10 and ffmpeg are needed first. On Ubuntu or Debian run:"
    echo "  sudo apt install python3.10 python3.10-venv ffmpeg libxcb-cursor0"
    echo "then run this again."
    exit 1
  fi
fi
echo "== using $PY"

echo "== SimBA environment"
[ -d .venv-simba ] || "$PY" -m venv .venv-simba
.venv-simba/bin/python -m pip install --upgrade pip wheel "setuptools<81"
.venv-simba/bin/python -m pip install -r env/requirements-simba.txt
.venv-simba/bin/python -m pip install -e .

echo "== DeepLabCut environment"
[ -d .venv-dlc ] || "$PY" -m venv .venv-dlc
.venv-dlc/bin/python -m pip install --upgrade pip wheel "setuptools<81"
if [ "$(uname)" = "Darwin" ]; then
  .venv-dlc/bin/python -m pip install torch==2.14.0 torchvision==0.29.0
else
  .venv-dlc/bin/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
fi
.venv-dlc/bin/python -m pip install -r env/requirements-dlc.txt

echo "== trained model"
.venv-simba/bin/python env/get_model.py

echo "== Done. Start the app with: bash start.sh"

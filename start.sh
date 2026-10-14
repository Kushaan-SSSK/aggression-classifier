#!/usr/bin/env bash
# Start the Aggression Scorer app on macOS or Linux: bash start.sh
cd "$(dirname "$0")"
if [ ! -x .venv-simba/bin/python ]; then
  echo "Not installed yet. Run: bash install.sh"
  exit 1
fi
exec .venv-simba/bin/python -m behavior_pipeline gui

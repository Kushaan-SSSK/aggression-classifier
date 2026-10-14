# Kushaan Sharma
"""Download the trained attack model from the GitHub release into models/.
The setup scripts run this; run it again by hand with: python env/get_model.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
URL = "https://github.com/Kushaan-SSSK/aggression-classifier/releases/download/v1.0.0/Attack_model.zip"
DEST = REPO / "models"


def main() -> int:
    if (DEST / "Attack.sav").exists():
        print(f"model already there: {DEST / 'Attack.sav'}")
        return 0
    DEST.mkdir(exist_ok=True)
    zip_path = DEST / "Attack_model.zip"
    print(f"downloading the attack model (about 45 MB) from\n  {URL}")

    shown = set()

    def progress(blocks, block_size, total):
        if total > 0:
            step = int(min(blocks * block_size / total, 1.0) * 10)
            if step not in shown:
                shown.add(step)
                print(f"  {step * 10}%", flush=True)

    try:
        urllib.request.urlretrieve(URL, zip_path, progress)
    except Exception as exc:
        print(f"download failed: {exc}\nCheck your internet connection and run: python env/get_model.py")
        return 1
    print("  unpacking")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DEST)
    zip_path.unlink()
    print(f"model saved in {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

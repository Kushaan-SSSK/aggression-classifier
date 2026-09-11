# Kushaan Sharma
"""Report which pipeline dependencies import in the current interpreter.
Run inside each venv: python env/check_env.py  (exit 0 if SimBA or DLC can run).
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


def _try(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return f"OK   {name:<22} {getattr(mod, '__version__', '')}"
    except Exception as exc:
        return f"MISS {name:<22} ({type(exc).__name__}: {str(exc)[:60]})"


def main() -> int:
    print(f"python  {sys.version.split()[0]}  ({sys.executable})")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # A winget-installed ffmpeg is only on PATH for shells opened after the install.
        from pathlib import Path

        hits = sorted((Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
        ffmpeg = str(hits[-1]) if hits else None
    if ffmpeg:
        ver = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        print(f"OK   ffmpeg                 {ver}")
    else:
        print("MISS ffmpeg                 (not on PATH - winget install Gyan.FFmpeg, then open a new shell)")

    simba_ok = dlc_ok = False
    print("-- SimBA role --")
    for name in ["simba", "numpy", "pandas", "sklearn", "cv2", "tables", "h5py", "shapely", "numba", "umap", "hdbscan", "yaml", "scipy"]:
        line = _try(name)
        print(line)
        if name == "simba" and line.startswith("OK"):
            simba_ok = True
    print("-- DeepLabCut role --")
    for name in ["torch", "torchvision", "deeplabcut"]:
        line = _try(name)
        print(line)
        if name == "deeplabcut" and line.startswith("OK"):
            dlc_ok = True
    try:
        import torch

        print(f"     torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception:
        pass
    print("-- pipeline package --")
    print(_try("behavior_pipeline"))
    print(f"\nSimBA role: {'ready' if simba_ok and ffmpeg else 'NOT ready'} | DLC role: {'ready' if dlc_ok else 'NOT ready'}")
    return 0 if (simba_ok or dlc_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

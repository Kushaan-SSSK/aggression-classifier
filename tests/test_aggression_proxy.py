# Kushaan Sharma
"""Aggression proxy heuristics on synthetic tracks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from behavior_pipeline.simba.aggression_proxy import DEFAULTS, merge_gaps, score_frames


def _tracks(n: int = 150, fps: float = 15.0, px_per_mm: float = 2.0) -> pd.DataFrame:
    # Animal 1 sits still; animal 2 approaches slowly, jerks about in contact for 50 frames, then walks off.
    t = np.arange(n)
    x1 = np.full(n, 200.0)
    x2 = np.where(t < 50, 365 - t * 3.0, np.where(t < 100, 215.0, 215.0 + (t - 100) * 2.0))
    y2 = np.where((t >= 50) & (t < 100), 100 + 40 * np.sin(t), 100.0)
    cols = {}
    for bp in ("Nose", "Center", "Tail_base"):
        cols[f"{bp}_1_x"], cols[f"{bp}_1_y"] = x1, np.full(n, 100.0)
        cols[f"{bp}_2_x"], cols[f"{bp}_2_y"] = x2, y2
    return pd.DataFrame(cols)


def test_merge_gaps():
    f = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=bool)
    out = merge_gaps(f, 3)
    assert out.tolist() == [True] * 6 + [False] * 4 + [True]
    assert merge_gaps(np.array([0, 0, 1], dtype=bool), 5).tolist() == [False, False, True]


def test_score_frames_flags_fast_contact_only():
    df = _tracks()
    scored = score_frames(df, px_per_mm=2.0, fps=15.0, p=dict(DEFAULTS))
    flags = scored["aggression_proxy"].to_numpy(dtype=bool)
    assert not flags[:40].any()
    assert flags[55:95].mean() > 0.8
    assert not flags[120:].any()
    assert scored["contact_dist_mm"].iloc[75] < 30 and scored["speed_animal2_mm_s"].iloc[75] > 150
    assert 0 <= scored["intensity"].max() <= 1

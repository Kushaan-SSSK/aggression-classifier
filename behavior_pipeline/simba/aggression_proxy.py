# Kushaan Sharma
"""Attack-like bouts from pose alone: close contact plus high centre speed.
A heuristic for ranking videos before a classifier exists; run as `aggression-proxy`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig
from .proximity import _px_per_mm_and_fps, bouts_from_flags, min_bout_filter

DEFAULTS: dict[str, Any] = {
    "contact_mm": 30.0,
    "speed_mm_s": 150.0,
    "smooth_ms": 200.0,
    "min_bout_ms": 100.0,
    "merge_gap_ms": 300.0,
    "pairs": [["Nose_1", "Center_2"], ["Nose_2", "Center_1"], ["Center_1", "Center_2"], ["Nose_1", "Tail_base_2"], ["Nose_2", "Tail_base_1"]],
    "inject_into_machine_results": True,
}


def merge_gaps(flags: np.ndarray, max_gap: int) -> np.ndarray:
    """Bridge False runs shorter than max_gap frames that sit between True runs."""
    flags = flags.astype(bool).copy()
    if max_gap <= 0 or flags.size == 0:
        return flags
    gaps = np.concatenate([[False], ~flags, [False]])
    edges = np.flatnonzero(np.diff(gaps.astype(int)))
    for start, end in zip(edges[::2], edges[1::2]):
        # Only interior gaps count; a leading or trailing run of False is not a gap.
        if 0 < start and end < flags.size and end - start < max_gap:
            flags[start:end] = True
    return flags


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.convolve(np.nan_to_num(x, nan=0.0), kernel, mode="same")


def score_frames(df: pd.DataFrame, px_per_mm: float, fps: float, p: dict[str, Any]) -> pd.DataFrame:
    """Per-frame contact distance, speeds, proxy flag and a 0-1 intensity score."""
    dist = np.full(len(df), np.inf)
    for a, b in p["pairs"]:
        if f"{a}_x" in df.columns and f"{b}_x" in df.columns:
            d = np.hypot(df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"]).to_numpy() / px_per_mm
            dist = np.fmin(dist, d)
    win = max(int(round(p["smooth_ms"] / 1000 * fps)), 1)
    speeds = []
    for a in ("Center_1", "Center_2"):
        x, y = df[f"{a}_x"].to_numpy(dtype=float), df[f"{a}_y"].to_numpy(dtype=float)
        v = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / px_per_mm * fps
        speeds.append(smooth(v, win))
    v_max = np.maximum(speeds[0], speeds[1])
    contact = dist < p["contact_mm"]
    fast = v_max > p["speed_mm_s"]
    raw = contact & fast
    flags = merge_gaps(raw, int(round(p["merge_gap_ms"] / 1000 * fps)))
    flags = min_bout_filter(flags, int(round(p["min_bout_ms"] / 1000 * fps)))
    intensity = np.clip(v_max / (2 * p["speed_mm_s"]), 0, 1) * contact
    return pd.DataFrame({"contact_dist_mm": dist, "speed_animal1_mm_s": speeds[0], "speed_animal2_mm_s": speeds[1],
                         "in_contact": contact.astype(int), "fast_contact_raw": raw.astype(int),
                         "aggression_proxy": flags.astype(int), "intensity": intensity}, index=df.index)


def compute_aggression_proxy(cfg: PipelineConfig, video_names: list[str] | None = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(cfg.get("aggression_proxy") or {})}
    pose_dir = cfg.simba_project_folder / "csv" / "outlier_corrected_movement_location"
    out_dir = cfg.simba_project_folder / "logs" / "aggression_proxy"
    out_dir.mkdir(parents=True, exist_ok=True)
    mr_dir = cfg.simba_project_folder / "csv" / "machine_results"
    calib = _px_per_mm_and_fps(cfg)
    files = sorted(pose_dir.glob("*.csv"))
    if video_names:
        files = [f for f in files if f.stem in set(video_names)]
    if not files:
        raise FileNotFoundError(f"no pose files in {pose_dir}; run the project stage first")
    summary = []
    for f in files:
        name = f.stem
        if name not in calib:
            LOG.warning("%s: not in video_info.csv, skipping", name)
            continue
        px_per_mm, fps = calib[name]
        df = pd.read_csv(f, index_col=0)
        scored = score_frames(df, px_per_mm, fps, p)
        scored.to_csv(out_dir / f"{name}.csv")
        flags = scored["aggression_proxy"].to_numpy(dtype=bool)
        bouts = bouts_from_flags(flags, fps)
        if len(bouts):
            bouts["peak_speed_mm_s"] = [float(np.max(np.maximum(scored["speed_animal1_mm_s"], scored["speed_animal2_mm_s"])[s:e])) for s, e in zip(bouts["start_frame"], bouts["end_frame"])]
            bouts = bouts.sort_values("start_frame").reset_index(drop=True)
        bouts.to_csv(out_dir / f"{name}_bouts.csv", index=False)
        minutes = len(df) / fps / 60
        summary.append({"video": name, "frames": len(df), "minutes": round(minutes, 2),
                        "contact_pct": round(float(scored["in_contact"].mean() * 100), 2),
                        "proxy_frames": int(flags.sum()), "proxy_pct": round(float(flags.mean() * 100), 2),
                        "n_bouts": int(len(bouts)), "bouts_per_min": round(len(bouts) / minutes, 2) if minutes else 0.0,
                        "total_bout_s": round(float(bouts["duration_s"].sum()), 2) if len(bouts) else 0.0,
                        "mean_bout_s": round(float(bouts["duration_s"].mean()), 2) if len(bouts) else 0.0,
                        "first_bout_s": round(float(bouts["start_s"].iloc[0]), 1) if len(bouts) else None})
        if p["inject_into_machine_results"]:
            mr = mr_dir / f"{name}.csv"
            if mr.exists():
                m = pd.read_csv(mr, index_col=0)
                if len(m) == len(flags):
                    m["Probability_Aggression_proxy"] = scored["intensity"].to_numpy()
                    m["Aggression_proxy"] = flags.astype(int)
                    m.to_csv(mr)
                else:
                    LOG.warning("%s: machine_results has %d rows but pose has %d; not injecting", name, len(m), len(flags))
            else:
                LOG.warning("%s: no machine_results file (run proximity first); not injecting", name)
        LOG.info("%s: %d attack-like bouts (%.1f/min), %.1f%% of frames in contact", name, len(bouts), summary[-1]["bouts_per_min"], summary[-1]["contact_pct"])
    table = pd.DataFrame(summary)
    table.to_csv(cfg.simba_project_folder / "logs" / "aggression_proxy_summary.csv", index=False)
    return table

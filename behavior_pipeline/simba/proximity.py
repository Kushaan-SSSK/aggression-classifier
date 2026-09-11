# Kushaan Sharma
"""Flag frames where the two mice are within a body-part distance threshold and
write per-video proximity bouts. Run as the `proximity` stage of the CLI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig


def _px_per_mm_and_fps(cfg: PipelineConfig) -> dict[str, tuple[float, float]]:
    info = cfg.simba_project_folder / "logs" / "video_info.csv"
    if not info.exists():
        raise FileNotFoundError(f"{info} missing; run the project stage first")
    df = pd.read_csv(info)
    return {str(r["Video"]): (float(r["pixels/mm"]), float(r["fps"])) for _, r in df.iterrows()}


def min_bout_filter(flags: np.ndarray, min_frames: int) -> np.ndarray:
    """Zero out True runs shorter than min_frames."""
    flags = flags.astype(bool).copy()
    if min_frames <= 1 or flags.size == 0:
        return flags
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_frames:
            flags[start:end] = False
    return flags


def bouts_from_flags(flags: np.ndarray, fps: float) -> pd.DataFrame:
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    recs = [{"start_frame": int(s), "end_frame": int(e), "start_s": s / fps, "end_s": e / fps, "duration_s": (e - s) / fps}
            for s, e in zip(edges[::2], edges[1::2])]
    return pd.DataFrame(recs, columns=["start_frame", "end_frame", "start_s", "end_s", "duration_s"])


def compute_proximity(cfg: PipelineConfig, video_names: list[str] | None = None) -> pd.DataFrame:
    pose_dir = cfg.simba_project_folder / "csv" / "outlier_corrected_movement_location"
    out_dir = cfg.simba_project_folder / "logs" / "proximity"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(p) for p in cfg.get("proximity.pairs", [["Nose_1", "Center_2"], ["Nose_2", "Center_1"], ["Center_1", "Center_2"]])]
    threshold_mm = float(cfg.get("proximity.threshold_mm", 40))
    min_bout_ms = float(cfg.get("proximity.min_bout_ms", 200))
    calib = _px_per_mm_and_fps(cfg)
    inject = bool(cfg.get("proximity.inject_into_machine_results", True))
    mr_dir = cfg.simba_project_folder / "csv" / "machine_results"

    files = sorted(pose_dir.glob("*.csv"))
    if video_names:
        files = [f for f in files if f.stem in set(video_names)]
    if not files:
        raise FileNotFoundError(f"no pose files in {pose_dir}")
    summary = []
    for f in files:
        name = f.stem
        if name not in calib:
            LOG.warning("%s: not in video_info.csv, skipping", name)
            continue
        px_per_mm, fps = calib[name]
        df = pd.read_csv(f, index_col=0)
        dist = pd.DataFrame(index=df.index)
        for a, b in pairs:
            dist[f"{a}-{b}_mm"] = np.hypot(df[f"{a}_x"] - df[f"{b}_x"], df[f"{a}_y"] - df[f"{b}_y"]) / px_per_mm
        raw = (dist.min(axis=1) < threshold_mm).to_numpy()
        min_frames = int(round(min_bout_ms / 1000 * fps))
        flags = min_bout_filter(raw, min_frames)
        dist["in_proximity"] = flags.astype(int)
        dist.to_csv(out_dir / f"{name}.csv")
        bouts = bouts_from_flags(flags, fps)
        bouts.to_csv(out_dir / f"{name}_bouts.csv", index=False)
        summary.append({"video": name, "frames": len(df), "proximity_frames": int(flags.sum()),
                        "proximity_pct": round(float(flags.mean() * 100), 2), "n_bouts": len(bouts),
                        "mean_bout_s": round(float(bouts["duration_s"].mean()), 2) if len(bouts) else 0.0})
        if inject:
            mr = mr_dir / f"{name}.csv"
            m = None
            if mr.exists():
                m = pd.read_csv(mr, index_col=0)
            else:
                # No classifier output yet: seed machine_results from the feature file.
                feat = cfg.simba_project_folder / "csv" / "features_extracted" / f"{name}.csv"
                if feat.exists():
                    m = pd.read_csv(feat, index_col=0)
                    for clf in cfg.classifiers:
                        m[f"Probability_{clf}"] = 0.0
                        m[clf] = 0
                    mr_dir.mkdir(parents=True, exist_ok=True)
                    LOG.info("%s: machine_results seeded from features_extracted", name)
                else:
                    LOG.warning("%s: neither machine_results nor features_extracted exists; not injecting", name)
            if m is not None:
                if len(m) == len(flags):
                    m["Probability_Proximity"] = flags.astype(float)
                    m["Proximity"] = flags.astype(int)
                    m.to_csv(mr)
                else:
                    LOG.warning("%s: machine_results has %d rows but pose has %d; not injecting", name, len(m), len(flags))
        LOG.info("%s: %.1f%% of frames in proximity, %d bouts", name, summary[-1]["proximity_pct"], len(bouts))
    table = pd.DataFrame(summary)
    table.to_csv(cfg.simba_project_folder / "logs" / "proximity_summary.csv", index=False)
    return table

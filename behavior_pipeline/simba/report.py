# Kushaan Sharma
"""Per-video behavioural endpoints from machine_results: latency to the first bout, bout and episode counts,
total and mean durations. Run as the `report` stage of the CLI; writes summary.csv, bouts.csv, results.xlsx and traces/<video>.csv
(time, probability and decision per behaviour, small enough for the app to plot).

Times count from the start of the processed clip, which is the intruder entry, so latency is time after entry.
`recording_start_s` / `recording_end_s` add the clip start back for finding a bout in the original recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) frame indices of each run of ones, end exclusive."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge_runs(bl: list[tuple[int, int]], gap_frames: int) -> list[tuple[int, int]]:
    """Join runs separated by at most gap_frames into episodes."""
    out: list[list[int]] = []
    for a, b in bl:
        if out and a - out[-1][1] <= gap_frames:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def video_endpoints(pred: np.ndarray, fps: float, episode_gap_s: float = 1.0) -> dict:
    """Endpoints of one binary trace; latency is None when nothing was scored."""
    pred = np.asarray(pred, dtype=bool)
    bl = runs(pred)
    dur = np.array([(b - a) / fps for a, b in bl])
    n = len(pred)
    return {"session_s": round(n / fps, 1),
            "latency_s": round(bl[0][0] / fps, 2) if bl else None,
            "bouts": len(bl),
            "episodes": len(merge_runs(bl, int(round(episode_gap_s * fps)))),
            "total_s": round(float(dur.sum()), 2),
            "pct_time": round(100.0 * pred.sum() / max(n, 1), 2),
            "mean_bout_s": round(float(dur.mean()), 2) if bl else None,
            "longest_bout_s": round(float(dur.max()), 2) if bl else None,
            "bouts_per_min": round(len(bl) / max(n / fps / 60.0, 1e-9), 2)}


def _manifest_groups(cfg: PipelineConfig) -> dict[str, dict]:
    """video_name -> {group, cohort} from the manifest, when it has those columns."""
    path = cfg.path("paths.video_manifest")
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = [c for c in ("group", "cohort") if c in df.columns]
    return {str(r["video_name"]): {c: r[c] for c in cols} for _, r in df.iterrows()}


def _clip_start(cfg: PipelineConfig, video: str) -> float:
    sidecar = cfg.path("paths.processed_videos") / f"{video}.json"
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        return float(meta.get("clip_start_s", meta.get("intruder_entry_s", 0.0)) or 0.0)
    return 0.0


def build_report(cfg: PipelineConfig, classifiers: list[str] | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Summary (one row per video and behaviour) and bout list from csv/machine_results."""
    from .annotations import _video_fps
    from .infer import trained_classifiers

    classifiers = classifiers or trained_classifiers(cfg)
    out_dir = Path(out_dir) if out_dir else cfg.root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = cfg.simba_project_folder / "csv" / "machine_results"
    files = sorted(results.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no scored videos in {results}; run the infer stage first")
    fps_by_video = _video_fps(cfg)
    groups = _manifest_groups(cfg)
    gap_s = float(cfg.get("evaluate.episode_gap_ms", 1000) or 0) / 1000.0
    summary, bout_rows = [], []
    (out_dir / "traces").mkdir(exist_ok=True)
    for f in files:
        v = f.stem
        cols = [c for clf in classifiers for c in (clf, f"Probability_{clf}")]
        df = pd.read_csv(f, usecols=lambda c: c in cols)
        fps = fps_by_video.get(v, cfg.fps)
        offset = _clip_start(cfg, v)
        trace = pd.DataFrame({"frame": np.arange(len(df)), "time_s": np.round(np.arange(len(df)) / fps, 3)})
        for clf in classifiers:
            if clf not in df.columns:
                continue
            pred = df[clf].to_numpy().astype(bool)
            prob = df[f"Probability_{clf}"].to_numpy(dtype=float) if f"Probability_{clf}" in df.columns else pred.astype(float)
            trace[f"Probability_{clf}"], trace[clf] = np.round(prob, 4), pred.astype(int)
            summary.append({"video": v, **groups.get(v, {}), "behavior": clf, **video_endpoints(pred, fps, gap_s)})
            for i, (a, b) in enumerate(runs(pred), start=1):
                bout_rows.append({"video": v, "behavior": clf, "bout": i, "start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                                  "duration_s": round((b - a) / fps, 2), "mean_prob": round(float(prob[a:b].mean()), 3),
                                  "peak_prob": round(float(prob[a:b].max()), 3), "start_frame": int(a), "end_frame": int(b),
                                  "recording_start_s": round(offset + a / fps, 2), "recording_end_s": round(offset + b / fps, 2)})
        trace.to_csv(out_dir / "traces" / f"{v}.csv", index=False)
    table = pd.DataFrame(summary)
    bouts = pd.DataFrame(bout_rows, columns=["video", "behavior", "bout", "start_s", "end_s", "duration_s", "mean_prob", "peak_prob",
                                             "start_frame", "end_frame", "recording_start_s", "recording_end_s"])
    table.to_csv(out_dir / "summary.csv", index=False)
    bouts.to_csv(out_dir / "bouts.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / "results.xlsx") as xl:
            table.to_excel(xl, sheet_name="summary", index=False)
            bouts.to_excel(xl, sheet_name="bouts", index=False)
    except (ImportError, PermissionError) as exc:  # no openpyxl, or the workbook is open in Excel
        LOG.warning("results.xlsx not written: %s", exc)
    LOG.info("report for %d video(s) -> %s", len(files), out_dir)
    return table

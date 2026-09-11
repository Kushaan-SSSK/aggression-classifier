# Kushaan Sharma
"""Turn the annotator's timestamp CSV into SimBA targets_inserted files.
Run as the `annotate` stage of the CLI; `--make-todo` writes the annotation sheet.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..config import LOG, PipelineConfig
from ..video.manifest import AnnotationRow, read_annotations


def _video_fps(cfg: PipelineConfig) -> dict[str, float]:
    info = cfg.simba_project_folder / "logs" / "video_info.csv"
    if not info.exists():
        return {}
    df = pd.read_csv(info)
    return {str(r["Video"]): float(r["fps"]) for _, r in df.iterrows()}


def build_targets(cfg: PipelineConfig, annotations_path: str | Path | None = None, features_dir: Path | None = None, targets_dir: Path | None = None) -> list[Path]:
    """Write one targets_inserted CSV per video and return the paths."""
    annotations_path = Path(annotations_path) if annotations_path else cfg.path("paths.annotations")
    features_dir = features_dir or (cfg.simba_project_folder / "csv" / "features_extracted")
    targets_dir = targets_dir or (cfg.simba_project_folder / "csv" / "targets_inserted")
    targets_dir.mkdir(parents=True, exist_ok=True)
    rows = read_annotations(annotations_path, cfg)
    classifiers = cfg.classifiers
    none_token = str(cfg.get("annotations.none_token", "NONE")).upper()
    fps_by_video = _video_fps(cfg)

    by_video: dict[str, list[AnnotationRow]] = defaultdict(list)
    for r in rows:
        by_video[r.video_name].append(r)

    feature_files = {p.stem: p for p in features_dir.glob("*.csv")}
    if not feature_files:
        raise FileNotFoundError(f"no feature files in {features_dir}; run the project stage first")

    videos = set(by_video)
    if cfg.get("annotations.include_unannotated_videos", False):
        videos |= set(feature_files)
    written: list[Path] = []
    summary = []
    for name in sorted(videos):
        if name not in feature_files:
            LOG.warning("%s: annotated but no features file; skipping", name)
            continue
        fps = fps_by_video.get(name, cfg.fps)
        df = pd.read_csv(feature_files[name], index_col=0)
        n = len(df)
        for clf in classifiers:
            df[clf] = 0
        counts = {clf: 0 for clf in classifiers}
        for r in by_video.get(name, []):
            if r.behavior == none_token:
                continue
            start = max(int(round(r.start_s * fps)), 0)
            end = min(int(round(r.end_s * fps)), n)
            if start >= n:
                LOG.warning("%s: %s bout at %.1fs starts after the video ends (%d frames)", name, r.behavior, r.start_s, n)
                continue
            df.iloc[start:end, df.columns.get_loc(r.behavior)] = 1
            counts[r.behavior] += max(end - start, 0)
        out = targets_dir / f"{name}.csv"
        df.to_csv(out)
        written.append(out)
        summary.append({"video": name, "frames": n, **{f"{c}_frames": counts[c] for c in classifiers}})
        LOG.info("%s: targets written (%s)", name, ", ".join(f"{c}={counts[c]}" for c in classifiers))
    if summary:
        pd.DataFrame(summary).to_csv(cfg.simba_project_folder / "logs" / "annotation_summary.csv", index=False)
    return written


def write_todo(cfg: PipelineConfig, out_path: str | Path | None = None) -> Path:
    """Write annotations_todo.csv with one TODO row per processed video."""
    import json

    out_path = Path(out_path) if out_path else cfg.path("paths.annotations").with_name("annotations_todo.csv")
    proc_dir = cfg.path("paths.processed_videos")
    rows = []
    for sidecar in sorted(proc_dir.glob("*.json")):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        out = meta.get("output", {})
        fps = float(out.get("fps", cfg.fps))
        dur = float(out.get("duration_s", 0.0))
        rows.append({
            "video_name": meta.get("video_name", sidecar.stem), "behavior": "TODO", "start_s": "", "end_s": "", "annotator": "",
            "notes": (f"clip is 0-{dur:.0f} s; 0 = intruder entry (raw video {meta.get('intruder_entry_s', 0):.0f} s); "
                      f"seconds = frame counter / {fps:g}; behaviours: Attack, Social_investigation, or NONE if none occur"),
        })
    df = pd.DataFrame(rows, columns=["video_name", "behavior", "start_s", "end_s", "annotator", "notes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOG.info("annotation todo sheet with %d videos -> %s", len(df), out_path)
    return out_path


def bouts_to_frames_table(rows: list[AnnotationRow], fps: float) -> pd.DataFrame:
    """Tabulate annotation rows as start/end frames."""
    recs = []
    for r in rows:
        if r.start_s is None:
            continue
        recs.append({"video_name": r.video_name, "behavior": r.behavior, "start_frame": int(round(r.start_s * fps)),
                     "end_frame": int(round(r.end_s * fps)), "duration_s": r.end_s - r.start_s})
    return pd.DataFrame(recs)

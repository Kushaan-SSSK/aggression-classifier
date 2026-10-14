# Kushaan Sharma
"""Extra per-frame features next to SimBA's: centred multi-scale windows, resident-relative geometry,
identity-free pair measures and pixel motion inside the two mice's box.
Run as the `features-extra` stage of the CLI; writes csv/features_extra/<video>.csv (one row per frame).

SimBA's 16-body-part windows are 2-15 frames and look only backwards, so at 15 fps the model never sees more
than the last second. These windows are centred and reach 4 s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig

BODY = ["Ear_left", "Ear_right", "Nose", "Center", "Lat_left", "Lat_right", "Tail_base"]
PREFIX = "ex_"


def extra_dir(cfg: PipelineConfig) -> Path:
    return cfg.simba_project_folder / "csv" / "features_extra"


def _xy(df: pd.DataFrame, bp: str, a: int) -> np.ndarray:
    return df[[f"{bp}_{a}_x", f"{bp}_{a}_y"]].to_numpy(float)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n > 1e-9, n, np.nan)


def _deriv(x: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(x, axis=0) * fps if len(x) > 1 else np.zeros_like(x)


def pose_signals(df: pd.DataFrame, px_per_mm: float, fps: float) -> pd.DataFrame:
    """Frame-level signals in mm, mm/s and cosines from a SimBA pose table (animal 1 = resident)."""
    s: dict[str, np.ndarray] = {}
    pts = {a: {bp: _xy(df, bp, a) / px_per_mm for bp in BODY} for a in (1, 2)}
    cen, head, vel, heading, length = {}, {}, {}, {}, {}
    for a in (1, 2):
        p = pts[a]
        cen[a] = (p["Center"] + p["Lat_left"] + p["Lat_right"]) / 3
        head[a] = (p["Nose"] + p["Ear_left"] + p["Ear_right"]) / 3
        heading[a] = _unit(head[a] - p["Tail_base"])
        length[a] = np.linalg.norm(p["Nose"] - p["Tail_base"], axis=1)
        vel[a] = _deriv(cen[a], fps)
        speed = np.linalg.norm(vel[a], axis=1)
        s[f"speed_{a}"] = speed
        s[f"nose_speed_{a}"] = np.linalg.norm(_deriv(p["Nose"], fps), axis=1)
        s[f"acc_{a}"] = np.linalg.norm(_deriv(vel[a], fps), axis=1)
        ang = np.unwrap(np.nan_to_num(np.arctan2(heading[a][:, 1], heading[a][:, 0])))
        s[f"angvel_{a}"] = np.abs(_deriv(ang, fps))
        s[f"length_{a}"] = length[a]
        s[f"width_{a}"] = np.linalg.norm(p["Lat_left"] - p["Lat_right"], axis=1)
        lik = df[[f"{bp}_{a}_p" for bp in BODY]].to_numpy(float)
        s[f"lik_mean_{a}"] = lik.mean(axis=1)
        s[f"lik_min_{a}"] = lik.min(axis=1)
    d12 = cen[2] - cen[1]
    u12 = _unit(d12)
    s["dist_centres"] = np.linalg.norm(d12, axis=1)
    s["dist_noses"] = np.linalg.norm(pts[1]["Nose"] - pts[2]["Nose"], axis=1)
    s["dist_nose1_tail2"] = np.linalg.norm(pts[1]["Nose"] - pts[2]["Tail_base"], axis=1)
    s["dist_nose2_tail1"] = np.linalg.norm(pts[2]["Nose"] - pts[1]["Tail_base"], axis=1)
    s["dist_nose1_centre2"] = np.linalg.norm(pts[1]["Nose"] - cen[2], axis=1)
    s["dist_nose2_centre1"] = np.linalg.norm(pts[2]["Nose"] - cen[1], axis=1)
    a1 = np.stack([pts[1][bp] for bp in BODY], axis=1)
    a2 = np.stack([pts[2][bp] for bp in BODY], axis=1)
    s["dist_min_bodyparts"] = np.linalg.norm(a1[:, :, None, :] - a2[:, None, :, :], axis=3).min(axis=(1, 2))
    # where each nose sits along the other's body: 0 tail base, 1 nose, <0 behind, lateral offset in body lengths
    for a, b in ((1, 2), (2, 1)):
        axis = pts[b]["Nose"] - pts[b]["Tail_base"]
        l2 = np.maximum((axis ** 2).sum(axis=1), 1e-6)
        rel = pts[a]["Nose"] - pts[b]["Tail_base"]
        along = (rel * axis).sum(axis=1) / l2
        lateral = np.abs(rel[:, 0] * axis[:, 1] - rel[:, 1] * axis[:, 0]) / l2
        s[f"nose{a}_along_{b}"] = np.clip(along, -2, 3)
        s[f"nose{a}_lateral_{b}"] = np.clip(lateral, 0, 2)
    s["facing_1"] = (heading[1] * u12).sum(axis=1)       # 1 = resident's head points at the intruder
    s["facing_2"] = -(heading[2] * u12).sum(axis=1)
    s["heading_align"] = (heading[1] * heading[2]).sum(axis=1)
    s["closing_speed"] = -_deriv(s["dist_centres"], fps)
    s["v1_toward_2"] = (vel[1] * u12).sum(axis=1)
    s["v2_away_from_1"] = (vel[2] * u12).sum(axis=1)
    moving = (s["speed_1"] > 20) & (s["speed_2"] > 20)
    s["velocity_align"] = np.where(moving, (_unit(vel[1]) * _unit(vel[2])).sum(axis=1), 0.0)
    lo1, hi1, lo2, hi2 = a1.min(axis=1), a1.max(axis=1), a2.min(axis=1), a2.max(axis=1)
    inter = np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None).prod(axis=1)
    area1, area2 = (hi1 - lo1).prod(axis=1), (hi2 - lo2).prod(axis=1)
    s["box_iou"] = inter / np.maximum(area1 + area2 - inter, 1e-6)
    s["speed_max"] = np.maximum(s["speed_1"], s["speed_2"])
    s["speed_min"] = np.minimum(s["speed_1"], s["speed_2"])
    s["acc_max"] = np.maximum(s["acc_1"], s["acc_2"])
    s["angvel_max"] = np.maximum(s["angvel_1"], s["angvel_2"])
    s["dist_nose_body_min"] = np.minimum(s["dist_nose1_centre2"], s["dist_nose2_centre1"])
    s["lik_min_both"] = np.minimum(s["lik_min_1"], s["lik_min_2"])
    out = pd.DataFrame(s)
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)


def pixel_signals(video: Path, df: pd.DataFrame, px_per_mm: float = 1.0, pad_px: int = 20,
                  ignore: tuple[int, int, int, int] | None = None, dark_threshold: int = 60) -> pd.DataFrame:
    """Frame-difference energy inside the union box of both mice and over the whole frame, and the dark blobs in the box
    (one blob when the mice touch). The box is the keypoints' extent plus pad_px, so it follows the mice even when
    their keypoints are wrong."""
    import cv2

    xs = df[[c for c in df.columns if c.endswith("_x") and not c.startswith("Tail_end")]].to_numpy(float)
    ys = df[[c for c in df.columns if c.endswith("_y") and not c.startswith("Tail_end")]].to_numpy(float)
    cap = cv2.VideoCapture(str(video))
    n = len(df)
    rec = np.zeros((n, 5))
    prev = None
    kernel = np.ones((5, 5), np.uint8)
    for t in range(n):
        ok, frame = cap.read()
        if not ok:
            rec[t:] = np.nan
            break
        g = frame[:, :, 0].astype(np.int16) if frame.ndim == 3 else frame.astype(np.int16)
        if ignore is not None:
            x0, y0, x1, y1 = ignore
            g[y0:y1, x0:x1] = 255
        h, w = g.shape
        x0 = int(np.clip(np.nanmin(xs[t]) - pad_px, 0, w - 1)) if np.isfinite(xs[t]).any() else 0
        x1 = int(np.clip(np.nanmax(xs[t]) + pad_px, x0 + 1, w)) if np.isfinite(xs[t]).any() else w
        y0 = int(np.clip(np.nanmin(ys[t]) - pad_px, 0, h - 1)) if np.isfinite(ys[t]).any() else 0
        y1 = int(np.clip(np.nanmax(ys[t]) + pad_px, y0 + 1, h)) if np.isfinite(ys[t]).any() else h
        if prev is not None:
            diff = np.abs(g - prev)
            rec[t, 0] = diff[y0:y1, x0:x1].mean()
            rec[t, 1] = diff.mean()
        dark = (g[y0:y1, x0:x1] < dark_threshold).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1] if cnt > 1 else np.array([0])
        rec[t, 2] = (areas > 0.25 * max(areas[0], 1)).sum() if areas[0] else 0
        rec[t, 3] = areas[0]
        rec[t, 4] = (x1 - x0) * (y1 - y0)
        prev = g
    cap.release()
    rec[0, :2] = rec[1, :2] if n > 1 else 0
    out = pd.DataFrame(rec, columns=["motion_box", "motion_frame", "dark_blobs_box", "dark_largest_px", "box_area_mm2"])
    out["box_area_mm2"] /= px_per_mm ** 2
    out["motion_ratio"] = out["motion_box"] / out["motion_frame"].replace(0, np.nan)
    out["dark_largest_rel"] = out["dark_largest_px"] / max(float(np.nanmedian(out["dark_largest_px"])), 1.0)
    return out.drop(columns=["dark_largest_px"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

HR_REGIONS = ["union", "res", "int", "head_res", "head_int", "flow_mean", "flow_p95", "flow_coherence"]
HR_COLUMNS = [f"hr_{k}_{s}" for k in HR_REGIONS for s in (("mean",) if k == "flow_coherence" else ("mean", "peak"))] + ["hr_int_over_res"]


def _raw_frames(cfg: PipelineConfig, video: str, fps: float):
    """Greyscale frames of the processed clip re-cut from its raw recording at `fps`, with the preprocess crop.

    Yields (frames_per_processed_frame, iterator of uint8 frames); None when the sidecar or source is missing.
    """
    import json
    import subprocess

    from ..video.preprocess import ffmpeg_path

    side = cfg.path("paths.processed_videos") / f"{video}.json"
    if not side.exists():
        return None
    meta = json.loads(side.read_text(encoding="utf-8"))
    src = Path(meta.get("source_path", ""))
    filters = list(meta.get("filters_pass1") or [])
    src_fps = float((meta.get("source") or {}).get("fps") or 0)
    if not src.exists() or not filters or meta.get("clip_start_s") is None:
        return None
    step = max(1, int(round(min(fps, src_fps) / cfg.fps))) if src_fps else 1
    out_fps = cfg.fps * step
    filters = [f"fps={out_fps:g}" if f.startswith("fps=") else f for f in filters]
    if "format=gray" not in filters:
        filters.append("format=gray")
    w = h = None
    for f in filters:
        if f.startswith("crop="):
            w, h = (int(v) for v in f[5:].split(":")[:2])
    if w is None:
        w, h = int(meta["source"]["width"]), int(meta["source"]["height"])
    cmd = [ffmpeg_path(cfg), "-loglevel", "error", "-ss", f"{float(meta['clip_start_s']):.3f}", "-t", f"{float(meta['clip_length_s']):.3f}",
           "-i", str(src), "-vf", ",".join(filters), "-an", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]

    def frames():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 8)
        try:
            while True:
                buf = proc.stdout.read(w * h)
                if len(buf) < w * h:
                    break
                yield np.frombuffer(buf, np.uint8).reshape(h, w)
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()

    return step, frames()


def _box(xs: np.ndarray, ys: np.ndarray, pad: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return None
    x0, x1 = int(max(xs[ok].min() - pad, 0)), int(min(xs[ok].max() + pad, w))
    y0, y1 = int(max(ys[ok].min() - pad, 0)), int(min(ys[ok].max() + pad, h))
    return (x0, y0, x1, y1) if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def hires_signals(cfg: PipelineConfig, video: str, df: pd.DataFrame, px_per_mm: float, fps: float = 30.0) -> pd.DataFrame | None:
    """Motion measured between frames `fps` apart (re-cut from the raw recording), folded back onto the
    processed frames as the mean and the peak of each frame's sub-steps.

    Regions: union box of both mice, each mouse's box, a 30 mm square on each nose. Optical flow (Farneback,
    half resolution) in the union box: mean speed, 95th-percentile speed, and coherence |mean vector| / mean speed
    (one direction for following, scattered for a scuffle). Speeds in mm/s.
    """
    import cv2

    got = _raw_frames(cfg, video, fps)
    if got is None:
        return None
    step, frames = got
    n = len(df)
    body = [bp for bp in BODY]
    xy = {a: (df[[f"{bp}_{a}_x" for bp in body]].to_numpy(float), df[[f"{bp}_{a}_y" for bp in body]].to_numpy(float)) for a in (1, 2)}
    nose = {a: df[[f"Nose_{a}_x", f"Nose_{a}_y"]].to_numpy(float) for a in (1, 2)}
    pad, head = 10 * px_per_mm, 15 * px_per_mm
    names = HR_REGIONS
    sub = np.full((n, step, len(names)), np.nan, np.float32)
    rate = cfg.fps * step / px_per_mm        # px per sub-step -> mm/s
    prev = None
    k = 0
    for g in frames:
        t, j = divmod(k, step)
        # sub-step j of processed frame t spans raw frames k-1 -> k; frame 0's first step has no predecessor
        if t >= n:
            break
        if prev is not None:
            h, w = g.shape
            diff = cv2.absdiff(g, prev)
            boxes = {"union": _box(np.r_[xy[1][0][t], xy[2][0][t]], np.r_[xy[1][1][t], xy[2][1][t]], pad, w, h),
                     "res": _box(xy[1][0][t], xy[1][1][t], pad, w, h), "int": _box(xy[2][0][t], xy[2][1][t], pad, w, h),
                     "head_res": _box(nose[1][t, :1], nose[1][t, 1:], head, w, h), "head_int": _box(nose[2][t, :1], nose[2][t, 1:], head, w, h)}
            for i, key in enumerate(names[:5]):
                b = boxes[key]
                if b is not None:
                    sub[t, j, i] = diff[b[1]:b[3], b[0]:b[2]].mean()
            b = boxes["union"]
            if b is not None:
                a0 = cv2.resize(prev[b[1]:b[3], b[0]:b[2]], None, fx=0.5, fy=0.5)
                a1 = cv2.resize(g[b[1]:b[3], b[0]:b[2]], None, fx=0.5, fy=0.5)
                if min(a0.shape) >= 8:
                    fl = cv2.calcOpticalFlowFarneback(a0, a1, None, 0.5, 2, 9, 2, 5, 1.1, 0) * 2.0
                    mag = np.hypot(fl[..., 0], fl[..., 1])
                    sub[t, j, 5] = mag.mean() * rate
                    sub[t, j, 6] = np.percentile(mag, 95) * rate
                    sub[t, j, 7] = np.hypot(fl[..., 0].mean(), fl[..., 1].mean()) / max(mag.mean(), 1e-6)
        prev = g
        k += 1
    cols = {}
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # frame 0 / frames without a box have no sub-steps
        for i, key in enumerate(names):
            cols[f"hr_{key}_mean"] = np.nanmean(sub[:, :, i], axis=1)
            if key != "flow_coherence":
                cols[f"hr_{key}_peak"] = np.nanmax(sub[:, :, i], axis=1)
    out = pd.DataFrame(cols)
    out["hr_int_over_res"] = out["hr_int_mean"] / out["hr_res_mean"].replace(0, np.nan)
    if k < n * step - step:
        LOG.warning("%s: %d raw frames at %g fps for %d processed frames", video, k, cfg.fps * step, n)
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)


def window_features(sig: pd.DataFrame, windows=(3, 7, 15, 31, 61), minmax=(7, 31), delta=(7, 15)) -> pd.DataFrame:
    """Centred rolling mean/std for every window, min/max for some, and future-minus-past means (onset / offset)."""
    cols: dict[str, pd.Series] = {}
    for c in sig.columns:
        x = sig[c]
        cols[c] = x
        for w in windows:
            r = x.rolling(w, center=True, min_periods=1)
            cols[f"{c}_mean{w}"] = r.mean()
            cols[f"{c}_std{w}"] = r.std().fillna(0)
            if w in minmax:
                cols[f"{c}_min{w}"] = r.min()
                cols[f"{c}_max{w}"] = r.max()
        for w in delta:
            past = x.rolling(w, min_periods=1).mean()
            future = x[::-1].rolling(w, min_periods=1).mean()[::-1]
            cols[f"{c}_delta{w}"] = future - past
    out = pd.DataFrame(cols)
    out.columns = [PREFIX + c for c in out.columns]
    return out.astype(np.float32)


def video_paths(cfg: PipelineConfig) -> dict[str, Path]:
    return {p.stem: p for p in cfg.path("paths.processed_videos").glob("*.mp4")}


def compute_extra_features(cfg: PipelineConfig, video_names: list[str] | None = None, overwrite: bool = False,
                           pixels: bool = True) -> list[Path]:
    from ..pose.blob import counter_rect
    from .proximity import _px_per_mm_and_fps

    pose_dir = cfg.simba_project_folder / "csv" / "outlier_corrected_movement_location"
    out_dir = extra_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = _px_per_mm_and_fps(cfg)
    videos = video_paths(cfg)
    files = sorted(pose_dir.glob("*.csv"))
    if video_names:
        files = [f for f in files if f.stem in set(video_names)]
    written = []
    for f in files:
        out = out_dir / f.name
        if out.exists() and not overwrite and out.stat().st_mtime > f.stat().st_mtime:
            continue
        if f.stem not in calib:
            LOG.warning("%s: not in video_info.csv, skipped", f.stem)
            continue
        px_per_mm, fps = calib[f.stem]
        df = pd.read_csv(f, index_col=0)
        sig = pose_signals(df, px_per_mm, fps)
        if pixels and f.stem in videos:
            import cv2

            cap = cv2.VideoCapture(str(videos[f.stem]))
            shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            cap.release()
            pix = pixel_signals(videos[f.stem], df, px_per_mm, pad_px=int(round(10 * px_per_mm)), ignore=counter_rect(shape, cfg))
            sig = pd.concat([sig, pix.set_index(sig.index)], axis=1)
        elif pixels:
            LOG.warning("%s: no processed video, pixel features left out", f.stem)
        hr_fps = float(cfg.get("features_extra.pixel_fps", 0) or 0)
        if pixels and hr_fps > cfg.fps:
            hr = hires_signals(cfg, f.stem, df, px_per_mm, hr_fps)
            if hr is None:
                LOG.warning("%s: no raw recording for %g fps motion, left as zeros", f.stem, hr_fps)
                hr = pd.DataFrame(0.0, index=range(len(sig)), columns=HR_COLUMNS)
            sig = pd.concat([sig, hr.set_index(sig.index)], axis=1)
        base = cfg.simba_project_folder / "csv" / "features_extra_base"
        base.mkdir(exist_ok=True)
        sig.astype(np.float32).to_csv(base / f.name, float_format="%.6g")
        feats = window_features(sig)
        feats.index = df.index
        feats.to_csv(out)
        written.append(out)
        LOG.info("%s: %d extra features -> %s", f.stem, feats.shape[1], out.name)
    return written


def load_extra(cfg: PipelineConfig, video: str) -> pd.DataFrame | None:
    p = extra_dir(cfg) / f"{video}.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


def append_to_features(cfg: PipelineConfig, video_names: list[str] | None = None) -> list[str]:
    """Replace the ex_ columns of csv/features_extracted/<video>.csv with the current extra features.

    SimBA's inference and the annotate stage read every non-pose column of these files, so the extra
    features reach training and inference without further changes. Re-run after SimBA's feature step.
    """
    feat_dir = cfg.simba_project_folder / "csv" / "features_extracted"
    done, mismatched = [], []
    for f in sorted(feat_dir.glob("*.csv")):
        if video_names and f.stem not in set(video_names):
            continue
        ex = load_extra(cfg, f.stem)
        if ex is None:
            LOG.warning("%s: no extra features, file left as SimBA wrote it", f.stem)
            continue
        df = pd.read_csv(f, index_col=0)
        df = df[[c for c in df.columns if not c.startswith(PREFIX)]]
        if len(ex) != len(df):
            # both come from the same pose file, so a difference means a half-written file or a second process
            LOG.error("%s: %d extra-feature rows for %d SimBA feature rows; file left unchanged", f.stem, len(ex), len(df))
            mismatched.append(f.stem)
            continue
        ex.index = df.index
        pd.concat([df, ex], axis=1).to_csv(f, float_format="%.6g")
        done.append(f.stem)
    LOG.info("extra features appended to %d feature file(s)", len(done))
    if mismatched:
        raise RuntimeError(f"row counts differ for {mismatched}; rerun `project --step features` (and check that no other "
                           "pipeline process is writing the project)")
    return done


def run_features_extra(cfg: PipelineConfig, video_names: list[str] | None = None, overwrite: bool = False) -> list[str]:
    compute_extra_features(cfg, video_names, overwrite=overwrite, pixels=bool(cfg.get("features_extra.pixels", True)))
    return append_to_features(cfg, video_names)

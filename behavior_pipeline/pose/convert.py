# Kushaan Sharma

"""Stage 2c: convert DeepLabCut H5 or CSV pose output to SimBA's "2 animals; 16 body-parts" CSV,
re-linking animal identities frame to frame and writing a qc.json per video.
Run with: python -m behavior_pipeline convert
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..config import LOG, SIMBA_16BP_COLUMNS, SIMBA_16BP_PER_ANIMAL, PipelineConfig


class ConvertError(ValueError):
    pass


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".h5", ".hdf5"):
        df = pd.read_hdf(path)
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as fh:
            first = [fh.readline().split(",")[0].strip().lower() for _ in range(4)]
        # multi-animal CSVs carry an extra "individuals" header row
        n_header = 4 if first[1] == "individuals" else 3
        df = pd.read_csv(path, header=list(range(n_header)), index_col=0)
    else:
        raise ConvertError(f"unsupported pose file type: {path}")
    if not isinstance(df.columns, pd.MultiIndex):
        raise ConvertError(f"{path}: expected DLC multi-index columns")
    return df


def _level(df: pd.DataFrame, name: str) -> int | None:
    names = [str(n).lower() if n is not None else "" for n in df.columns.names]
    return names.index(name) if name in names else None


def read_pose(path: str | Path, keypoint_map: dict[str, str], n_animals: int = 2) -> tuple[np.ndarray, list[str]]:
    """Return an array [frames, animals, 8, 3] of x, y, likelihood in SimBA body-part order."""
    path = Path(path)
    df = _read_any(path)
    df = df.sort_index()
    lvl_ind = _level(df, "individuals")
    lvl_bp = _level(df, "bodyparts")
    lvl_co = _level(df, "coords")
    if lvl_bp is None or lvl_co is None:
        raise ConvertError(f"{path}: columns must have 'bodyparts' and 'coords' levels, got {df.columns.names}")

    if lvl_ind is not None:
        individuals = [str(i) for i in df.columns.get_level_values(lvl_ind).unique()]
    else:
        individuals = ["single"]
    missing_map = [v for v in keypoint_map.values() if v not in set(df.columns.get_level_values(lvl_bp))]
    bodyparts_available = sorted(set(str(b) for b in df.columns.get_level_values(lvl_bp)))

    n_frames = len(df)
    out = np.full((n_frames, n_animals, len(SIMBA_16BP_PER_ANIMAL), 3), np.nan, dtype=float)
    out[..., 2] = 0.0

    def _get(ind: str | None, bp: str, coord: str) -> np.ndarray | None:
        mask = (df.columns.get_level_values(lvl_bp) == bp) & (df.columns.get_level_values(lvl_co) == coord)
        if ind is not None and lvl_ind is not None:
            mask &= df.columns.get_level_values(lvl_ind) == ind
        cols = df.columns[mask]
        if len(cols) == 0:
            return None
        return df[cols[0]].to_numpy(dtype=float)

    if lvl_ind is not None:
        if len(individuals) < n_animals:
            LOG.warning("%s: only %d individual(s) in file, expected %d", path.name, len(individuals), n_animals)
        for a in range(min(n_animals, len(individuals))):
            for k, simba_bp in enumerate(SIMBA_16BP_PER_ANIMAL):
                src_bp = keypoint_map.get(simba_bp)
                if src_bp is None:
                    continue
                for c, coord in enumerate(("x", "y", "likelihood")):
                    arr = _get(individuals[a], src_bp, coord)
                    if arr is not None:
                        out[:, a, k, c] = arr
    else:
        # single-animal layout: body parts may carry _1/_2 suffixes
        for a in range(n_animals):
            for k, simba_bp in enumerate(SIMBA_16BP_PER_ANIMAL):
                src_bp = keypoint_map.get(simba_bp, simba_bp)
                candidates = [f"{src_bp}_{a + 1}", f"{simba_bp}_{a + 1}"] + ([src_bp] if n_animals == 1 else [])
                for cand in candidates:
                    arr_x = _get(None, cand, "x")
                    if arr_x is not None:
                        out[:, a, k, 0] = arr_x
                        out[:, a, k, 1] = _get(None, cand, "y")
                        lk = _get(None, cand, "likelihood")
                        out[:, a, k, 2] = lk if lk is not None else 1.0
                        break
    if np.isnan(out[..., 0]).all():
        raise ConvertError(
            f"{path.name}: none of the mapped body parts were found. Map={keypoint_map}. "
            f"Available body parts: {bodyparts_available[:40]}"
        )
    if missing_map:
        LOG.warning("%s: mapped body parts not in file: %s", path.name, missing_map)
    return out, individuals


def _anchor(frame: np.ndarray, anchor_idx: int, min_lik: float) -> np.ndarray:
    """Anchor point per animal: the anchor body part, else the mean of the valid parts."""
    pts = np.full((frame.shape[0], 2), np.nan)
    for a in range(frame.shape[0]):
        valid = frame[a, :, 2] >= min_lik
        valid &= ~np.isnan(frame[a, :, 0])
        if valid[anchor_idx]:
            pts[a] = frame[a, anchor_idx, :2]
        elif valid.any():
            pts[a] = frame[a, valid, :2].mean(axis=0)
    return pts


def link_identities(data: np.ndarray, anchor_bp: str = "Center", max_jump_px: float = 150.0, min_likelihood: float = 0.1) -> tuple[np.ndarray, dict[str, Any]]:
    """Reorder animals per frame so each track is spatially continuous; return (data, stats)."""
    n_frames, n_animals = data.shape[0], data.shape[1]
    anchor_idx = SIMBA_16BP_PER_ANIMAL.index(anchor_bp) if anchor_bp in SIMBA_16BP_PER_ANIMAL else 3
    out = data.copy()
    last = np.full((n_animals, 2), np.nan)
    stats = {"n_reassigned": 0, "n_jumps": 0, "n_frames_no_anchor": 0, "first_tracked_frame": None}
    big = 1e6
    for t in range(n_frames):
        pts = _anchor(data[t], anchor_idx, min_likelihood)
        have = ~np.isnan(pts[:, 0])
        if not have.any():
            stats["n_frames_no_anchor"] += 1
            continue
        if np.isnan(last[:, 0]).all():
            last = pts.copy()
            stats["first_tracked_frame"] = t
            continue
        # rows are previous tracks, columns are current detections
        cost = np.full((n_animals, n_animals), big)
        for i in range(n_animals):
            for j in range(n_animals):
                if have[j] and not np.isnan(last[i, 0]):
                    cost[i, j] = np.hypot(*(pts[j] - last[i]))
                elif have[j]:
                    cost[i, j] = big / 2
        rows, cols = linear_sum_assignment(cost)
        order = np.arange(n_animals)
        for i, j in zip(rows, cols):
            order[i] = j
        if not np.array_equal(order, np.arange(n_animals)):
            stats["n_reassigned"] += 1
        out[t] = data[t][order]
        new_pts = pts[order]
        for i in range(n_animals):
            if have[order[i]]:
                if not np.isnan(last[i, 0]) and np.hypot(*(new_pts[i] - last[i])) > max_jump_px:
                    stats["n_jumps"] += 1
                last[i] = new_pts[i]
    return out, stats


def to_simba_dataframe(data: np.ndarray, scorer: str = "behavior_pipeline") -> pd.DataFrame:
    """Lay the array out as a DLC-style CSV frame with SimBA's 16 body-part columns."""
    n_frames = data.shape[0]
    cols = []
    values = np.empty((n_frames, len(SIMBA_16BP_COLUMNS) * 3), dtype=float)
    c = 0
    for a in range(2):
        for k, bp in enumerate(SIMBA_16BP_PER_ANIMAL):
            name = f"{bp}_{a + 1}"
            for ci, coord in enumerate(("x", "y", "likelihood")):
                cols.append((scorer, name, coord))
                values[:, c] = data[:, a, k, ci] if a < data.shape[1] else np.nan
                c += 1
    columns = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    df = pd.DataFrame(values, columns=columns)
    lik_cols = [col for col in df.columns if col[2] == "likelihood"]
    df[lik_cols] = df[lik_cols].fillna(0.0)
    return df


def qc_stats(data: np.ndarray, min_likelihood: float) -> dict[str, Any]:
    """Per body part missing fraction and mean likelihood, plus centre tracking rates."""
    n_frames = data.shape[0]
    stats: dict[str, Any] = {"n_frames": int(n_frames), "per_body_part": {}}
    for a in range(data.shape[1]):
        for k, bp in enumerate(SIMBA_16BP_PER_ANIMAL):
            lik = data[:, a, k, 2]
            missing = np.isnan(data[:, a, k, 0]) | (lik < min_likelihood)
            stats["per_body_part"][f"{bp}_{a + 1}"] = {
                "pct_missing_or_low_conf": round(float(missing.mean() * 100), 2),
                "mean_likelihood": round(float(np.nanmean(lik)), 3) if n_frames else None,
            }
    c = SIMBA_16BP_PER_ANIMAL.index("Center")
    center_ok = (~np.isnan(data[:, :, c, 0])) & (data[:, :, c, 2] >= min_likelihood)
    stats["pct_frames_animal_center_tracked"] = {f"Animal_{a + 1}": round(float(center_ok[:, a].mean() * 100), 2) for a in range(data.shape[1])}
    stats["pct_frames_both_animals_have_center"] = round(float(center_ok.all(axis=1).mean() * 100), 2) if data.shape[1] >= 2 else None
    return stats


def convert_pose_file(src: str | Path, video_name: str, cfg: PipelineConfig, out_dir: Path | None = None, swap: bool = False,
                      keypoint_set: str | None = None, link: bool = True, extra_qc: dict[str, Any] | None = None) -> Path:
    """Convert one pose file and return the SimBA CSV path."""
    src = Path(src)
    out_dir = out_dir or cfg.path("paths.pose_simba")
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = keypoint_set or ("superanimal" if cfg.get("pose.backend", "superanimal") == "superanimal" else "lab_dlc")
    kp_map = cfg.get(f"pose.keypoint_map.{backend}")
    if not kp_map:
        raise ConvertError(f"no keypoint map for {backend!r} in config pose.keypoint_map")
    n_animals = int(cfg.get("simba.animal_cnt", 2))
    ident = cfg.get("pose.identity") or {}
    min_lik = float(ident.get("min_likelihood", 0.1))

    data, individuals = read_pose(src, kp_map, n_animals=n_animals)
    stats: dict[str, Any] = {}
    if link and n_animals > 1 and str(ident.get("method", "hungarian")) != "none":
        data, stats = link_identities(data, anchor_bp=str(ident.get("anchor_body_part", "Center")),
                                      max_jump_px=float(ident.get("max_jump_px", 150)), min_likelihood=min_lik)
    if swap and data.shape[1] >= 2:
        data = data[:, ::-1].copy()
    df = to_simba_dataframe(data)
    out_csv = out_dir / f"{video_name}.csv"
    df.to_csv(out_csv)
    qc = {"source": str(src), "video_name": video_name, "individuals_in_file": individuals, "keypoint_set": backend,
          "swapped": swap, "identity_linking": stats, **qc_stats(data, min_lik), **(extra_qc or {})}
    try:
        from ..video.preprocess import load_sidecar

        expected = int(load_sidecar(cfg, video_name)["output"]["nb_frames"])
        qc["n_frames_video"] = expected
        if expected != len(df):
            LOG.warning("%s: pose file has %d frames but the processed video has %d", video_name, len(df), expected)
    except Exception:
        # no sidecar when converting a single file outside the pipeline
        pass
    (out_dir / f"{video_name}.qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    LOG.info("%s: wrote %s (%d frames; reassigned=%s jumps=%s; both centres tracked %.1f%%)", video_name, out_csv.name, len(df),
             stats.get("n_reassigned"), stats.get("n_jumps"), qc.get("pct_frames_both_animals_have_center") or 0.0)
    return out_csv


def find_pose_file(video_name: str, pose_dir: Path) -> Path | None:
    """Pick the pose file for a video: tracked maDLC output first, then the newest H5, then CSV."""
    cands = [p for p in pose_dir.glob(f"{video_name}*.h5") if p.name.startswith(video_name)]
    # ELS58 must not pick up ELS580's files
    cands = [p for p in cands if p.name[len(video_name):len(video_name) + 1] in ("_", ".", "-", "")]
    for suffix in ("_el.h5", "_bx.h5", "_sk.h5"):
        for c in cands:
            if c.name.endswith(suffix):
                return c
    if cands:
        newest = max(cands, key=lambda p: p.stat().st_mtime)
        if len(cands) > 1:
            LOG.info("%s: %d pose files, using the newest: %s", video_name, len(cands), newest.name)
        return newest
    csvs = sorted(p for p in pose_dir.glob(f"{video_name}*.csv") if not p.name.endswith(".qc.json"))
    return csvs[0] if csvs else None


def resolve_swap(cfg: PipelineConfig, name: str, swap_names: set[str] | None, use_resident: bool = True) -> tuple[bool, str, dict[str, Any] | None]:
    """Return (swap, source, evidence); explicit --swap names win over the resident stage."""
    if swap_names and name in swap_names:
        return True, "cli", None
    if use_resident:
        from .resident import load_resident_decision

        dec = load_resident_decision(cfg, name)
        if dec is not None:
            evidence = {k: dec.get(k) for k in ("swap", "confidence", "d_animal1_px", "d_animal2_px", "n_frames_used", "reason", "pct_single_blob_pre_entry")}
            if dec.get("swap") is None:
                LOG.warning("%s: resident undecided (%s); Animal_1 may be the intruder, check the preview and use --swap", name, dec.get("reason"))
                return False, "none", evidence
            return bool(dec["swap"]), "resident_json", evidence
    return False, "none", None


def convert_all(cfg: PipelineConfig, video_names: list[str] | None = None, swap_names: set[str] | None = None,
                use_resident: bool = True) -> list[Path]:
    """Convert every processed video that has a pose file."""
    pose_dir = cfg.path("paths.pose_raw")
    if video_names is None:
        video_names = sorted({p.stem for p in cfg.path("paths.processed_videos").glob("*.mp4")})
    outputs = []
    unresolved: list[str] = []
    for name in video_names:
        src = find_pose_file(name, pose_dir)
        if src is None:
            LOG.warning("%s: no pose file in %s, skipping", name, pose_dir)
            continue
        mp4 = cfg.path("paths.processed_videos") / f"{name}.mp4"
        if mp4.exists() and src.stat().st_mtime < mp4.stat().st_mtime:
            LOG.warning("%s: pose file %s is older than the processed video; rerun the pose stage?", name, src.name)
        swap, source, evidence = resolve_swap(cfg, name, swap_names, use_resident=use_resident)
        if source == "none":
            unresolved.append(name)
        outputs.append(convert_pose_file(src, name, cfg, swap=swap, extra_qc={"swap_source": source, "resident": evidence}))
    if unresolved:
        LOG.warning("resident identity not established for %s (Animal_1 assumed resident)", unresolved)
    return outputs

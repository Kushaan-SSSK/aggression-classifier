# Kushaan Sharma

"""Resident identification: decides which tracked animal was in the cage before the intruder.

Run in .venv-simba:  python -m behavior_pipeline resident [--videos ...] [--preview]
Writes <paths.pose_resident>/<video>.resident.json, which the convert stage reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import LOG, SIMBA_16BP_PER_ANIMAL, PipelineConfig
from .blob import blob_params, boxes_for_video, build_backgrounds
from .convert import find_pose_file, link_identities, read_pose

DEFAULTS = {"max_match_px": 60.0, "min_confidence": 0.3, "min_frames": 10, "min_single_blob_frac": 0.5, "max_jump_px": 80.0,
            "fallback_window_s": 10.0, "fallback_frames": 15, "fallback_max_match_px": 150.0}


def resident_track(boxes: list[list[tuple[int, int, int, int]]], entry_frame: int, max_jump_px: float) -> tuple[np.ndarray, float]:
    """[n, 2] centre track of the animal present before entry_frame, and the fraction of pre-entry frames with one blob."""
    n = len(boxes)
    centres = [[(x + w / 2.0, y + h / 2.0) for x, y, w, h in b] for b in boxes]
    pre = list(range(min(entry_frame, n)))
    single = [t for t in pre if len(centres[t]) == 1]
    frac = len(single) / len(pre) if pre else 0.0
    track = np.full((n, 2), np.nan)
    if not single:
        return track, frac
    anchor = single[-1]
    last = np.array(centres[anchor][0])
    track[anchor] = last
    for t in range(anchor + 1, n):
        if centres[t]:
            pts = np.array(centres[t])
            d = np.hypot(*(pts - last).T)
            j = int(np.argmin(d))
            if d[j] <= max_jump_px:
                last = pts[j]
        # the last position is carried when nothing acceptable is seen, e.g. the hand in the frame
        track[t] = last
    for t in range(anchor - 1, -1, -1):
        if centres[t]:
            pts = np.array(centres[t])
            d = np.hypot(*(pts - track[t + 1]).T)
            j = int(np.argmin(d))
            track[t] = pts[j] if d[j] <= max_jump_px else track[t + 1]
        else:
            track[t] = track[t + 1]
    return track, frac


def mouse_like_boxes(clip: Path, boxes: list[list[tuple[int, int, int, int]]], max_center_intensity: float = 45.0,
                     darkest_only_until: int = 0) -> tuple[list[list[tuple[int, int, int, int]]], int]:
    """Drop boxes whose central region is not mouse-dark; before darkest_only_until keep only the darkest box."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    kept: list[list[tuple[int, int, int, int]]] = []
    n_rejected = 0
    for t, frame_boxes in enumerate(boxes):
        ok, fr = cap.read()
        if not ok:
            kept.append(list(frame_boxes))
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        scored = []
        for x, y, w, h in frame_boxes:
            # mean intensity of the central half of the box
            cx0, cy0 = x + w // 4, y + h // 4
            patch = g[cy0:cy0 + max(h // 2, 1), cx0:cx0 + max(w // 2, 1)]
            scored.append((float(patch.mean()) if patch.size else 255.0, (x, y, w, h)))
        good = [b for m, b in scored if m <= max_center_intensity]
        if t < darkest_only_until and good:
            good = [min((m, b) for m, b in scored if m <= max_center_intensity)[1]]
        n_rejected += len(frame_boxes) - len(good)
        kept.append(good)
    cap.release()
    return kept, n_rejected


def decide_swap(resident_xy: np.ndarray, animal1_xy: np.ndarray, animal2_xy: np.ndarray, max_match_px: float = 60.0,
                min_confidence: float = 0.3, min_frames: int = 10) -> dict[str, Any]:
    """Compare the resident track with both linked animal tracks (all [n, 2], NaN = missing)."""
    n = min(len(resident_xy), len(animal1_xy), len(animal2_xy))
    r, a1, a2 = resident_xy[:n], animal1_xy[:n], animal2_xy[:n]
    valid = np.isfinite(r).all(axis=1) & np.isfinite(a1).all(axis=1) & np.isfinite(a2).all(axis=1)
    out: dict[str, Any] = {"n_frames_used": int(valid.sum()), "d_animal1_px": None, "d_animal2_px": None, "confidence": None, "swap": None, "reason": ""}
    if valid.sum() < min_frames:
        out["reason"] = f"only {int(valid.sum())} frames with resident + both animals tracked (< {min_frames})"
        return out
    d1 = float(np.median(np.hypot(*(a1[valid] - r[valid]).T)))
    d2 = float(np.median(np.hypot(*(a2[valid] - r[valid]).T)))
    out.update({"d_animal1_px": round(d1, 1), "d_animal2_px": round(d2, 1)})
    conf = abs(d1 - d2) / max(d1, d2, 1e-9)
    out["confidence"] = round(float(conf), 3)
    if min(d1, d2) > max_match_px:
        out["reason"] = f"neither animal is near the resident track (min {min(d1, d2):.0f}px > {max_match_px}px)"
        return out
    if conf < min_confidence:
        out["reason"] = f"margin too small (confidence {conf:.2f} < {min_confidence})"
        return out
    out["swap"] = bool(d2 < d1)
    out["reason"] = "animal2 matches the resident track" if out["swap"] else "animal1 matches the resident track"
    return out


def _load_clip_meta(cfg: PipelineConfig, video_name: str) -> tuple[Path, dict[str, Any]]:
    clip_dir = cfg.path("paths.resident_clips", "data/resident_clips")
    clip = clip_dir / f"{video_name}.mp4"
    meta_path = clip_dir / f"{video_name}.json"
    if not clip.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{video_name}: no resident clip in {clip_dir} (enable video.resident_clip and rerun preprocess)")
    return clip, json.loads(meta_path.read_text(encoding="utf-8"))


def identify_resident(cfg: PipelineConfig, video_name: str, preview: bool = False) -> dict[str, Any]:
    """Decide the resident for one video and write its resident.json."""
    p = {**DEFAULTS, **(cfg.get("pose.resident") or {})}
    clip, meta = _load_clip_meta(cfg, video_name)
    entry_frame = int(meta.get("entry_frame", round(float(meta.get("pre_roll_s", 3)) * cfg.fps)))
    params = blob_params(cfg)
    # the clip is too short for its own background (a resting mouse would be part of it), so use the main video's
    main_mp4 = cfg.path("paths.processed_videos") / f"{video_name}.mp4"
    backgrounds = None
    if main_mp4.exists():
        bgs, _ = build_backgrounds(main_mp4, params)
        backgrounds = ([bgs[0]], 10**9)
    boxes, bstats = boxes_for_video(clip, params, cfg, backgrounds=backgrounds)
    boxes, n_rejected = mouse_like_boxes(clip, boxes, max_center_intensity=float(p.get("max_center_intensity", 45.0)),
                                         darkest_only_until=entry_frame)
    bstats["boxes_rejected_not_dark"] = n_rejected
    track, single_frac = resident_track(boxes, entry_frame, float(p["max_jump_px"]))

    pose_file = find_pose_file(video_name, cfg.path("paths.pose_raw"))
    if pose_file is None:
        raise FileNotFoundError(f"{video_name}: no pose file in {cfg.path('paths.pose_raw')}; run the pose stage first")
    kp_map = cfg.get("pose.keypoint_map.superanimal" if cfg.get("pose.backend", "superanimal") == "superanimal" else "pose.keypoint_map.lab_dlc")
    ident = cfg.get("pose.identity") or {}
    min_lik = float(ident.get("min_likelihood", 0.1))
    data, _ = read_pose(pose_file, kp_map, n_animals=2)
    linked, _ = link_identities(data, anchor_bp=str(ident.get("anchor_body_part", "Center")),
                                max_jump_px=float(ident.get("max_jump_px", 150)), min_likelihood=min_lik)
    c = SIMBA_16BP_PER_ANIMAL.index("Center")
    n_post = min(len(track) - entry_frame, len(linked))
    a = np.full((2, max(n_post, 0), 2), np.nan)
    for i in range(2):
        seg = linked[:n_post, i, c]
        ok = (seg[:, 2] >= min_lik) & np.isfinite(seg[:, 0])
        a[i, ok] = seg[ok, :2]
    res = track[entry_frame:entry_frame + n_post]
    decision = decide_swap(res, a[0], a[1], max_match_px=float(p["max_match_px"]), min_confidence=float(p["min_confidence"]),
                           min_frames=int(p["min_frames"]))
    method = "blob track vs linked Center over the post-entry clip"
    if decision["swap"] is not None and single_frac < float(p["min_single_blob_frac"]):
        decision["swap"] = None
        decision["reason"] = f"only {single_frac:.0%} of pre-entry frames show a single animal; entry time may be wrong"
    if decision["swap"] is None and entry_frame > 0 and np.isfinite(track[entry_frame - 1]).all() and single_frac >= float(p["min_single_blob_frac"]):
        # fallback: last pre-entry position against the first frames in which both mice are tracked
        last = track[entry_frame - 1]
        n_win = int(float(p["fallback_window_s"]) * cfg.fps)
        ok = np.zeros(min(n_win, len(linked)), dtype=bool)
        cent = np.full((2, len(ok), 2), np.nan)
        for i in range(2):
            seg = linked[:len(ok), i, c]
            good = (seg[:, 2] >= min_lik) & np.isfinite(seg[:, 0])
            cent[i, good] = seg[good, :2]
        both = np.flatnonzero(np.isfinite(cent[0, :, 0]) & np.isfinite(cent[1, :, 0]))[: int(p["fallback_frames"])]
        if len(both) >= 3:
            fb = decide_swap(np.tile(last, (len(both), 1)), cent[0, both], cent[1, both], max_match_px=float(p["fallback_max_match_px"]),
                             min_confidence=float(p["min_confidence"]), min_frames=3)
            fb["reason"] = f"fallback (first frames {int(both[0])}-{int(both[-1])} with both mice tracked vs last pre-entry position): {fb['reason']}"
            decision = fb
            method = "last pre-entry blob position vs first both-tracked linked Centers"
    result = {
        "video_name": video_name, "swap": decision["swap"], "confidence": decision["confidence"],
        "d_animal1_px": decision["d_animal1_px"], "d_animal2_px": decision["d_animal2_px"], "n_frames_used": decision["n_frames_used"],
        "reason": decision["reason"], "pct_single_blob_pre_entry": round(single_frac * 100, 1), "entry_frame": entry_frame,
        "clip": str(clip), "pose_file": str(pose_file), "method": method,
        "clip_boxes": {k: bstats[k] for k in ("n_frames", "frames_0_boxes", "frames_1_box", "frames_2_boxes")},
        "resident_track": [[round(float(x), 1), round(float(y), 1)] if np.isfinite(x) else None for x, y in track],
    }
    out_dir = cfg.path("paths.pose_resident", "data/pose/resident")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{video_name}.resident.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if preview:
        _write_preview(clip, track, entry_frame, out_dir / f"{video_name}.resident_preview.mp4")
    LOG.info("%s: resident = %s (d1=%s d2=%s conf=%s, %s)", video_name,
             "undecided" if result["swap"] is None else ("Animal_2 -> swap" if result["swap"] else "Animal_1"),
             result["d_animal1_px"], result["d_animal2_px"], result["confidence"], result["reason"])
    return result


def _write_preview(clip: Path, track: np.ndarray, entry_frame: int, out_mp4: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok or t >= len(track):
            break
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        if np.isfinite(track[t, 0]):
            cv2.circle(fr, (int(track[t, 0]), int(track[t, 1])), 8, (0, 0, 255), 2)
        cv2.putText(fr, "pre-entry" if t < entry_frame else "post-entry", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        wr.write(fr)
        t += 1
    cap.release()
    wr.release()


def load_resident_decision(cfg: PipelineConfig, video_name: str) -> dict[str, Any] | None:
    path = cfg.path("paths.pose_resident", "data/pose/resident") / f"{video_name}.resident.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_resident(cfg: PipelineConfig, video_names: list[str] | None = None, preview: bool = False) -> list[dict[str, Any]]:
    """Identify the resident for each video (default: every processed video) and warn about undecided ones."""
    if video_names is None:
        video_names = sorted({p.stem for p in cfg.path("paths.processed_videos").glob("*.mp4")})
    results = []
    for name in video_names:
        try:
            results.append(identify_resident(cfg, name, preview=preview))
        except FileNotFoundError as exc:
            LOG.warning("%s", exc)
    undecided = [r["video_name"] for r in results if r["swap"] is None]
    if undecided:
        LOG.warning("resident undecided for %s: check the preview and use `convert --swap <name>` if Animal_1 is the intruder", undecided)
    return results

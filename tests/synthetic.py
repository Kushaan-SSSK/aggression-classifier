# Kushaan Sharma
"""Synthetic video, pose H5 and annotations so the pipeline can be exercised without real data.
Run: python tests/synthetic.py [out_dir]  (or call make_synthetic_project from a test).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INTRUDER_ENTRY_S = 5.0
RAW_FPS = 30
W, H = 640, 480
CROP = (40, 20, 560, 440)
VIDEO_NAME = "synthetic_pair"

SUPERANIMAL_BODYPARTS = [
    "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip", "left_eye", "right_eye", "neck", "mid_back",
    "mouse_center", "mid_backend", "mid_backend2", "mid_backend3", "tail_base", "tail1", "tail2", "tail3", "tail4",
    "tail5", "left_shoulder", "left_midside", "left_hip", "right_shoulder", "right_midside", "right_hip", "tail_end",
    "head_midpoint",
]
# Keypoint offsets (along heading, across heading) in px for a roughly 70 px mouse.
_OFFSETS = {
    "nose": (35, 0), "left_ear": (22, -10), "right_ear": (22, 10), "left_ear_tip": (24, -14), "right_ear_tip": (24, 14),
    "left_eye": (28, -5), "right_eye": (28, 5), "neck": (15, 0), "mid_back": (5, 0), "mouse_center": (0, 0),
    "mid_backend": (-6, 0), "mid_backend2": (-12, 0), "mid_backend3": (-18, 0), "tail_base": (-25, 0),
    "tail1": (-32, 2), "tail2": (-40, 4), "tail3": (-48, 6), "tail4": (-56, 8), "tail5": (-64, 10),
    "left_shoulder": (12, -12), "left_midside": (0, -14), "left_hip": (-15, -12), "right_shoulder": (12, 12),
    "right_midside": (0, 14), "right_hip": (-15, 12), "tail_end": (-72, 12), "head_midpoint": (26, 0),
}


def _trajectories(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Return [frames, 2 animals, 3] (x, y, heading) inside the crop, with periodic approaches."""
    x0, y0, cw, ch = CROP
    pos = np.zeros((n_frames, 2, 3))
    pos[0, 0, :2] = (x0 + cw * 0.3, y0 + ch * 0.5)
    pos[0, 1, :2] = (x0 + cw * 0.7, y0 + ch * 0.5)
    vel = rng.normal(0, 1.5, size=(2, 2))
    for t in range(1, n_frames):
        # Every other 6 s block starts with 2 s of mutual attraction, giving close bouts.
        attract = ((t // (RAW_FPS * 6)) % 2 == 1) and (t % (RAW_FPS * 6)) < RAW_FPS * 2
        for a in range(2):
            vel[a] += rng.normal(0, 0.6, size=2)
            if attract:
                other = pos[t - 1, 1 - a, :2]
                vel[a] += 0.15 * (other - pos[t - 1, a, :2]) / (np.linalg.norm(other - pos[t - 1, a, :2]) + 1e-6) * 10
            speed = np.linalg.norm(vel[a])
            if speed > 6:
                vel[a] *= 6 / speed
            p = pos[t - 1, a, :2] + vel[a]
            for k, (lo, hi) in enumerate(((x0 + 40, x0 + cw - 40), (y0 + 40, y0 + ch - 40))):
                if p[k] < lo or p[k] > hi:
                    vel[a, k] *= -1
                    p[k] = np.clip(p[k], lo, hi)
            pos[t, a, :2] = p
            pos[t, a, 2] = np.arctan2(vel[a, 1], vel[a, 0])
    return pos


def _draw_frame(t: int, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    frame = np.full((H, W, 3), 60, dtype=np.uint8)
    x0, y0, cw, ch = CROP
    cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + ch), (150, 160, 170), -1)
    noise = rng.integers(0, 25, size=(ch, cw, 1), dtype=np.uint8)
    frame[y0:y0 + ch, x0:x0 + cw] = np.clip(frame[y0:y0 + ch, x0:x0 + cw].astype(int) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + ch), (90, 90, 90), 6)
    n_animals = 2 if t >= INTRUDER_ENTRY_S * RAW_FPS else 1
    for a in range(n_animals):
        x, y, heading = pos[t, a]
        cv2.ellipse(frame, (int(x), int(y)), (36, 16), float(np.degrees(heading)), 0, 360, (30, 30, 30), -1)
        nose = (int(x + 35 * np.cos(heading)), int(y + 35 * np.sin(heading)))
        cv2.circle(frame, nose, 5, (20, 20, 20), -1)
    return frame


def make_raw_video(path: Path, seconds: float, rng: np.random.Generator) -> np.ndarray:
    import cv2

    n_frames = int(seconds * RAW_FPS)
    pos = _trajectories(n_frames, rng)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), RAW_FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a VideoWriter (mp4v)")
    for t in range(n_frames):
        writer.write(_draw_frame(t, pos, rng))
    writer.release()
    return pos


def processed_frame_positions(pos: np.ndarray, out_fps: float, clip_len_s: float) -> np.ndarray:
    """Positions of the processed video's frames, in processed-video pixel coordinates."""
    n_raw = pos.shape[0]
    n_out = int(min(clip_len_s, n_raw / RAW_FPS - INTRUDER_ENTRY_S) * out_fps)
    raw_idx = np.minimum((INTRUDER_ENTRY_S * RAW_FPS + np.arange(n_out) * RAW_FPS / out_fps).astype(int), n_raw - 1)
    out = pos[raw_idx].copy()
    out[:, :, 0] -= CROP[0]
    out[:, :, 1] -= CROP[1]
    return out


def make_superanimal_h5(path: Path, proc_pos: np.ndarray, rng: np.random.Generator, swap_every: int = 97) -> None:
    n = proc_pos.shape[0]
    scorer = "DLC_superanimal_topviewmouse_hrnet_w32"
    individuals = ["individual1", "individual2"]
    cols = pd.MultiIndex.from_product([[scorer], individuals, SUPERANIMAL_BODYPARTS, ["x", "y", "likelihood"]],
                                      names=["scorer", "individuals", "bodyparts", "coords"])
    data = np.zeros((n, len(cols)))
    for t in range(n):
        order = [0, 1]
        if (t // swap_every) % 2 == 1:  # deliberate identity swaps for the linker to undo
            order = [1, 0]
        c = 0
        for a in order:
            x, y, heading = proc_pos[t, a]
            ca, sa = np.cos(heading), np.sin(heading)
            for bp in SUPERANIMAL_BODYPARTS:
                dx, dy = _OFFSETS[bp]
                px = x + dx * ca - dy * sa + rng.normal(0, 1.0)
                py = y + dx * sa + dy * ca + rng.normal(0, 1.0)
                lik = float(np.clip(rng.normal(0.9, 0.08), 0, 1))
                if rng.random() < 0.02:
                    px, py, lik = np.nan, np.nan, 0.0
                data[t, c:c + 3] = (px, py, lik)
                c += 3
    df = pd.DataFrame(data, columns=cols)
    df.to_hdf(path, key="df_with_missing", mode="w", format="table")


def make_annotations(path: Path, proc_pos: np.ndarray, out_fps: float, attack_px: float = 70, sniff_px: float = 130) -> pd.DataFrame:
    d = np.linalg.norm(proc_pos[:, 0, :2] - proc_pos[:, 1, :2], axis=1)
    rows = []

    def _bouts(mask: np.ndarray, label: str, min_frames: int) -> None:
        padded = np.concatenate([[False], mask, [False]])
        edges = np.flatnonzero(np.diff(padded.astype(int)))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_frames:
                rows.append({"video_name": VIDEO_NAME, "behavior": label, "start_s": round(s / out_fps, 3),
                             "end_s": round(e / out_fps, 3), "annotator": "synthetic", "notes": ""})

    _bouts(d < attack_px, "Attack", int(0.3 * out_fps))
    _bouts((d >= attack_px) & (d < sniff_px), "Social_investigation", int(0.3 * out_fps))
    df = pd.DataFrame(rows, columns=["video_name", "behavior", "start_s", "end_s", "annotator", "notes"])
    df.to_csv(path, index=False)
    return df


def make_synthetic_project(root: Path, template_config: Path, seconds: float = 45.0, seed: int = 0, n_estimators: int = 50) -> Path:
    """Create a self-contained project under root and return the new config path."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for sub in ("config", "data/raw_videos", "data/processed_videos", "data/pose/raw", "data/pose/simba_16bp", "data/annotations"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    with open(template_config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["train"]["rf_n_estimators"] = n_estimators
    cfg["train"]["evaluation"].update({"feature_importance_bar_graph": False, "precision_recall_curve": False, "model_meta_data_file": False})
    cfg["unsupervised"]["umap"]["n_neighbors"] = [5]
    cfg["unsupervised"]["hdbscan"]["min_cluster_size"] = [3]
    cfg["unsupervised"]["min_bout_length"] = 2
    cfg["pose"]["superanimal"]["create_labeled_video"] = False
    cfg_path = root / "config" / "pipeline.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    rng = np.random.default_rng(seed)
    raw = root / "data" / "raw_videos" / f"{VIDEO_NAME}.mp4"
    pos = make_raw_video(raw, seconds, rng)
    np.save(root / "data" / "raw_videos" / f"{VIDEO_NAME}_truth.npy", pos)

    pd.DataFrame([{"video_name": VIDEO_NAME, "source_path": f"data/raw_videos/{VIDEO_NAME}.mp4", "intruder_entry_s": INTRUDER_ENTRY_S,
                   "crop_x": CROP[0], "crop_y": CROP[1], "crop_w": CROP[2], "crop_h": CROP[3], "downsample": "",
                   "cage_width_mm": 290, "cohort": "synthetic", "lighting": "n/a", "notes": "generated"}]).to_csv(
        root / "data" / "annotations" / "video_manifest.csv", index=False)

    out_fps = float(cfg["video"]["fps"])
    proc = processed_frame_positions(pos, out_fps, float(cfg["video"]["clip_length_s"]))
    make_superanimal_h5(root / "data" / "pose" / "raw" / f"{VIDEO_NAME}_DLC_superanimal.h5", proc, rng)
    make_annotations(root / "data" / "annotations" / "annotations.csv", proc, out_fps)
    return cfg_path


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_run").resolve()
    tpl = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
    print(make_synthetic_project(out, tpl))

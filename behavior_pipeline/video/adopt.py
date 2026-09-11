# Kushaan Sharma

"""Stage 1b: adopt videos already processed by SimBA's batch tool, restoring the crop aspect
ratio and writing the same sidecars and manifest the preprocess stage produces.
Run with: python -m behavior_pipeline adopt --video-dir <folder>
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import LOG, PipelineConfig
from .manifest import safe_video_name
from .preprocess import PreprocessError, ffmpeg_path, probe_video


def _hms_to_s(txt: str) -> float:
    parts = [float(p) for p in str(txt).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def load_batch_log(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("video_data", {})


def find_video(video_dir: Path, stem: str) -> Path | None:
    hits = [p for p in video_dir.glob(f"{stem}*") if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]
    exact = [p for p in hits if p.stem == stem]
    return (exact or hits or [None])[0]


def adopt_video(stem: str, entry: dict[str, Any], src: Path, cfg: PipelineConfig, out_dir: Path, overwrite: bool = False) -> dict[str, Any]:
    """Copy or rescale one batch-tool video into the processed folder and write its sidecar."""
    name = safe_video_name(stem)
    out_path = out_dir / f"{name}.mp4"
    sidecar = out_dir / f"{name}.json"
    crop = entry.get("crop_settings") or {}
    info = probe_video(src, cfg)
    if entry.get("crop") and crop:
        target_w, target_h = int(crop["width"]), int(crop["height"])
        crop_tuple = (int(crop["top_left_x"]), int(crop["top_left_y"]), target_w, target_h)
    else:
        target_w, target_h = info["width"], info["height"]
        crop_tuple = None
    target_w, target_h = target_w // 2 * 2, target_h // 2 * 2
    # the batch tool scales the crop back up to the source size, so undo that
    stretched = (info["width"], info["height"]) != (target_w, target_h)
    if out_path.exists() and sidecar.exists() and not overwrite:
        LOG.info("%s: already adopted, skipping", name)
        return json.loads(sidecar.read_text(encoding="utf-8"))
    t0 = time.time()
    if stretched:
        LOG.info("%s: restoring aspect ratio %dx%d -> %dx%d", name, info["width"], info["height"], target_w, target_h)
        cmd = [ffmpeg_path(cfg), "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={target_w}:{target_h}",
               "-an", "-c:v", "libx264", "-crf", str(cfg.get("video.output_crf", 18)), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise PreprocessError(f"{name}: ffmpeg failed: {res.stderr[-1500:]}")
    else:
        shutil.copy2(src, out_path)
    out_info = probe_video(out_path, cfg)
    clip = entry.get("clip_settings") or {}
    entry_s = _hms_to_s(clip.get("start", "0")) if entry.get("clip") else 0.0
    cage_mm = cfg.get("simba.cage_width_mm")
    px_per_mm = cfg.get("simba.px_per_mm") or (round(out_info["width"] / float(cage_mm), 4) if cage_mm else None)
    meta = {
        "video_name": name,
        "original_name": stem,
        "source_path": str(src),
        "adopted_from_simba_batch": True,
        "batch_log_entry": entry,
        "source": info,
        "intruder_entry_s": entry_s,
        "clip_length_s": _hms_to_s(clip.get("stop", "0")) - entry_s if entry.get("clip") else out_info["duration_s"],
        "crop": crop_tuple,
        "aspect_restored": stretched,
        "output": {"path": str(out_path), "width": out_info["width"], "height": out_info["height"], "fps": out_info["fps"],
                   "nb_frames": out_info["nb_frames"], "duration_s": out_info["duration_s"]},
        "cage_width_mm": cage_mm,
        "px_per_mm": px_per_mm,
        "notes": "adopted from SimBA batch output; grayscale/CLAHE/frame counter already applied by SimBA",
        "elapsed_s": round(time.time() - t0, 1),
    }
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("%s: %dx%d @ %.2f fps, %d frames, entry at %.0fs, px/mm=%s", name, out_info["width"], out_info["height"],
             out_info["fps"], out_info["nb_frames"], entry_s, px_per_mm)
    return meta


def write_manifest_from_log(entries: dict[str, dict[str, Any]], cfg: PipelineConfig, path: Path, names: set[str] | None = None,
                            source_dir: Path | None = None, exclude: tuple[str, ...] = (), overwrite: bool = False,
                            note: str = "") -> Path:
    """Write a video manifest from a SimBA batch_process_log.json."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True (--overwrite) to replace it")
    rows = []
    for stem, e in entries.items():
        name = safe_video_name(stem)
        if names is not None and name not in names:
            continue
        if any(tok.lower() in stem.lower() for tok in exclude):
            LOG.info("%s: excluded from the manifest", stem)
            continue
        crop = e.get("crop_settings") or {}
        clip = e.get("clip_settings") or {}
        vi = e.get("video_info", {})
        src = str(vi.get("file_path", ""))
        if source_dir is not None:
            src = str(Path(source_dir) / Path(src.replace("\\", "/")).name)
        geometry = f"source {vi.get('width')}x{vi.get('height')} @ {float(vi.get('fps', 0)):.0f} fps"
        rows.append({
            "video_name": name,
            "source_path": src,
            "intruder_entry_s": _hms_to_s(clip.get("start", "0")) if e.get("clip") else 0,
            "crop_x": crop.get("top_left_x", ""), "crop_y": crop.get("top_left_y", ""),
            "crop_w": crop.get("width", ""), "crop_h": crop.get("height", ""),
            "downsample": "", "cage_width_mm": cfg.get("simba.cage_width_mm", ""),
            "cohort": stem.split(".ELS")[0] if ".ELS" in stem else "", "lighting": "",
            "notes": (note or "crop/entry from SimBA batch_process_log.json") + "; " + geometry,
        })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.csv")
        shutil.copy2(path, backup)
        LOG.info("previous manifest backed up to %s", backup.name)
    df.to_csv(path, index=False)
    LOG.info("manifest with %d rows -> %s", len(df), path)
    return path


def adopt_all(cfg: PipelineConfig, video_dir: Path, batch_log: Path | None = None, exclude_tokens: tuple[str, ...] = ("do_not_use",), overwrite: bool = False) -> dict[str, dict[str, Any]]:
    """Adopt every video listed in the batch log and write the manifest; return the sidecars by name."""
    video_dir = Path(video_dir)
    batch_log = Path(batch_log) if batch_log else (video_dir / "batch_process_log.json")
    if not batch_log.exists():
        cand = list(video_dir.parent.glob("batch_process_log.json"))
        if not cand:
            raise FileNotFoundError(f"batch_process_log.json not found in {video_dir} or its parent")
        batch_log = cand[0]
    entries = load_batch_log(batch_log)
    out_dir = cfg.path("paths.processed_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    adopted: dict[str, dict[str, Any]] = {}
    for stem, entry in entries.items():
        src = find_video(video_dir, stem)
        if src is None:
            LOG.warning("%s: listed in the batch log but no video found in %s", stem, video_dir)
            continue
        if any(tok.lower() in src.name.lower() for tok in exclude_tokens):
            LOG.warning("%s: skipped (%s)", src.name, "name marks it as unusable")
            continue
        meta = adopt_video(stem, entry, src, cfg, out_dir, overwrite=overwrite)
        adopted[meta["video_name"]] = meta
    write_manifest_from_log(entries, cfg, cfg.path("paths.video_manifest"), names=set(adopted), overwrite=True,
                            note="from SimBA batch_process_log.json; source_path is the lab's original location")
    return adopted

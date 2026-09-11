# Kushaan Sharma

"""Stage 1: crop, trim, downsample, grayscale, 15 fps, CLAHE and frame counter per manifest row.
Writes <processed_videos>/<video_name>.mp4 plus a JSON sidecar of the parameters used.
Run with: python -m behavior_pipeline preprocess
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import LOG, PipelineConfig
from .manifest import VideoEntry


class PreprocessError(RuntimeError):
    pass


def ffmpeg_path(cfg: PipelineConfig | None = None) -> str:
    binary = cfg.get("video.ffmpeg_binary", "ffmpeg") if cfg is not None else "ffmpeg"
    found = shutil.which(binary)
    if found:
        return found
    # winget installs ffmpeg here but does not always add it to PATH
    local = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for cand in sorted(local.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")):
        return str(cand)
    raise PreprocessError("ffmpeg not found. Install with: winget install Gyan.FFmpeg (then open a new shell)")


def ffprobe_path(cfg: PipelineConfig | None = None) -> str:
    ff = Path(ffmpeg_path(cfg))
    probe = ff.with_name("ffprobe" + ff.suffix)
    if not probe.exists():
        found = shutil.which("ffprobe")
        if not found:
            raise PreprocessError("ffprobe not found next to ffmpeg")
        return found
    return str(probe)


def probe_video(path: str | Path, cfg: PipelineConfig | None = None) -> dict[str, Any]:
    """Return width, height, fps, duration_s and nb_frames of the first video stream."""
    cmd = [
        ffprobe_path(cfg), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise PreprocessError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    info = json.loads(out.stdout)
    stream = info["streams"][0]

    def _rate(txt: str) -> float:
        num, _, den = txt.partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    fps = _rate(stream.get("avg_frame_rate", "0/1")) or _rate(stream.get("r_frame_rate", "0/1"))
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    nb = stream.get("nb_frames")
    nb_frames = int(nb) if nb and str(nb).isdigit() else int(round(duration * fps))
    return {"width": int(stream["width"]), "height": int(stream["height"]), "fps": fps,
            "duration_s": duration, "nb_frames": nb_frames}


def _run(cmd: list[str]) -> None:
    LOG.debug("RUN %s", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise PreprocessError(f"command failed ({out.returncode}): {' '.join(cmd)}\n{out.stderr[-2000:]}")


def _pass1_filters(entry: VideoEntry, cfg: PipelineConfig, src_info: dict[str, Any], grayscale: bool | None = None) -> tuple[list[str], int, int]:
    """Build the ffmpeg -vf chain; return (filters, out_w, out_h)."""
    filters: list[str] = []
    w, h = src_info["width"], src_info["height"]
    if entry.crop is not None:
        x, y, cw, ch = entry.crop
        if x + cw > w or y + ch > h:
            raise PreprocessError(f"{entry.video_name}: crop {entry.crop} exceeds frame {w}x{h}")
        filters.append(f"crop={cw}:{ch}:{x}:{y}")
        w, h = cw, ch
    target_w = entry.downsample_width
    if target_w is None and cfg.get("video.downsample.enabled", False):
        target_w = int(cfg.get("video.downsample.max_width", 640))
    if target_w and target_w < w:
        new_h = int(round(h * target_w / w / 2) * 2)
        target_w = int(target_w // 2 * 2)
        filters.append(f"scale={target_w}:{new_h}")
        w, h = target_w, new_h
    # libx264 needs even dimensions
    if w % 2 or h % 2:
        w, h = w // 2 * 2, h // 2 * 2
        filters.append(f"crop={w}:{h}:0:0")
    filters.append(f"fps={cfg.fps}")
    if grayscale is None:
        grayscale = bool(cfg.get("video.grayscale", True))
    if grayscale:
        filters.append("format=gray")
    return filters, w, h


def _draw_counter(frame: np.ndarray, idx: int, position: str, font_scale: float) -> None:
    import cv2

    text = f"Frame: {idx}"
    thickness = 2
    colour = frame.ndim == 3
    black = (0, 0, 0) if colour else 0
    white = (255, 255, 255) if colour else 255
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    h, w = frame.shape[:2]
    margin = 8
    x = margin if "left" in position else max(w - tw - margin, 0)
    y = th + margin if "top" in position else h - margin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, black, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv2.LINE_AA)


def _apply_clahe(frame: np.ndarray, clahe) -> np.ndarray:
    import cv2

    if frame.ndim == 2:
        return clahe.apply(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _pass2_opencv(tmp_path: Path, out_path: Path, cfg: PipelineConfig, width: int, height: int,
                  colour: bool = False, clahe_on: bool | None = None, counter_on: bool | None = None) -> int:
    """Apply CLAHE and the frame counter, piping frames into ffmpeg; return the frame count."""
    import cv2

    clahe_on = bool(cfg.get("video.clahe.enabled", True)) if clahe_on is None else clahe_on
    counter_on = bool(cfg.get("video.frame_counter.enabled", True)) if counter_on is None else counter_on
    clahe = None
    if clahe_on:
        grid = tuple(int(v) for v in cfg.get("video.clahe.tile_grid", [16, 16]))
        clahe = cv2.createCLAHE(clipLimit=float(cfg.get("video.clahe.clip_limit", 2.0)), tileGridSize=grid)
    position = str(cfg.get("video.frame_counter.position", "top_left"))
    font_scale = float(cfg.get("video.frame_counter.font_scale", 0.7))

    cap = cv2.VideoCapture(str(tmp_path))
    if not cap.isOpened():
        raise PreprocessError(f"OpenCV could not open {tmp_path}")
    cmd = [
        ffmpeg_path(cfg), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24" if colour else "gray", "-s", f"{width}x{height}", "-r", str(cfg.fps), "-i", "-",
        "-an", "-c:v", str(cfg.get("video.output_codec", "libx264")), "-crf", str(cfg.get("video.output_crf", 18)),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if colour and frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif not colour and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            if clahe is not None:
                frame = _apply_clahe(frame, clahe)
            if counter_on:
                _draw_counter(frame, n, position, font_scale)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            n += 1
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        rc = proc.wait()
    if rc != 0:
        raise PreprocessError(f"ffmpeg encode failed: {err[-2000:]}")
    return n


def params_hash(entry: VideoEntry, cfg: PipelineConfig, start_s: float, clip_len_s: float,
                grayscale: bool, clahe_on: bool, counter_on: bool) -> str:
    """Short digest of every setting that affects the output frames."""
    payload = {
        "source": str(entry.source_path), "crop": entry.crop, "downsample": entry.downsample_width,
        "start_s": round(float(start_s), 3), "clip_len_s": round(float(clip_len_s), 3), "fps": cfg.fps,
        "grayscale": grayscale, "clahe": (cfg.get("video.clahe") if clahe_on else None), "counter": counter_on,
        "downsample_cfg": cfg.get("video.downsample"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def preprocess_video(entry: VideoEntry, cfg: PipelineConfig, out_dir: Path | None = None, overwrite: bool = False,
                     start_s: float | None = None, clip_len_s: float | None = None, name: str | None = None,
                     frame_counter: bool | None = None, extra_meta: dict[str, Any] | None = None) -> Path:
    """Preprocess one manifest row and return the output mp4 path."""
    out_dir = out_dir or cfg.path("paths.processed_videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or entry.video_name
    out_path = out_dir / f"{name}.mp4"
    sidecar = out_dir / f"{name}.json"
    start = entry.intruder_entry_s if start_s is None else max(float(start_s), 0.0)
    clip_len = float(cfg.get("video.clip_length_s", 300)) if clip_len_s is None else float(clip_len_s)
    grayscale = bool(cfg.get("video.grayscale", True))
    clahe_on = bool(cfg.get("video.clahe.enabled", True))
    counter_on = bool(cfg.get("video.frame_counter.enabled", True)) if frame_counter is None else bool(frame_counter)
    digest = params_hash(entry, cfg, start, clip_len, grayscale, clahe_on, counter_on)

    if out_path.exists() and sidecar.exists() and not overwrite:
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        if old.get("adopted_from_simba_batch") or old.get("params_hash") != digest:
            raise PreprocessError(
                f"{name}: {out_path.name} exists but was made with different settings "
                f"(adopted={bool(old.get('adopted_from_simba_batch'))}, hash {old.get('params_hash')} != {digest}); "
                "rerun with --overwrite or move the old output away"
            )
        LOG.info("%s: already processed with the same settings, skipping", name)
        return out_path
    if not entry.source_path.exists():
        raise PreprocessError(f"{name}: source video missing: {entry.source_path}")

    t0 = time.time()
    src_info = probe_video(entry.source_path, cfg)
    if start >= src_info["duration_s"]:
        raise PreprocessError(
            f"{name}: clip start {start}s is beyond the video duration {src_info['duration_s']:.1f}s"
        )
    filters, w, h = _pass1_filters(entry, cfg, src_info, grayscale=grayscale)
    need_pass2 = clahe_on or counter_on
    pass1_out = out_dir / f"{name}.pass1.mp4" if need_pass2 else out_path
    # the intermediate file is re-encoded once more, so keep it near lossless
    crf = "12" if need_pass2 else str(cfg.get("video.output_crf", 18))
    cmd = [
        ffmpeg_path(cfg), "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{clip_len:.3f}", "-i", str(entry.source_path),
        "-vf", ",".join(filters), "-an", "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p", str(pass1_out),
    ]
    LOG.info("%s: pass 1 (ffmpeg) start=%.1fs len=%.0fs filters=%s", name, start, clip_len, ",".join(filters))
    _run(cmd)
    n_frames: int
    if need_pass2:
        LOG.info("%s: pass 2 (%s%s%s)", name, "CLAHE" if clahe_on else "", " + " if clahe_on and counter_on else "",
                 "frame counter" if counter_on else "")
        n_frames = _pass2_opencv(pass1_out, out_path, cfg, w, h, colour=not grayscale, clahe_on=clahe_on, counter_on=counter_on)
        pass1_out.unlink(missing_ok=True)
    else:
        n_frames = probe_video(out_path, cfg)["nb_frames"]

    cage_mm = entry.cage_width_mm or cfg.get("simba.cage_width_mm")
    px_per_mm = cfg.get("simba.px_per_mm")
    if px_per_mm is None and cage_mm:
        px_per_mm = round(w / float(cage_mm), 4)
    meta = {
        "video_name": name,
        "source_path": str(entry.source_path),
        "source": src_info,
        "intruder_entry_s": entry.intruder_entry_s,
        "clip_start_s": start,
        "clip_length_s": clip_len,
        "crop": entry.crop,
        "downsample_width": entry.downsample_width,
        "filters_pass1": filters,
        "grayscale": grayscale,
        "clahe": {**(cfg.get("video.clahe") or {}), "enabled": clahe_on},
        "frame_counter": {**(cfg.get("video.frame_counter") or {}), "enabled": counter_on},
        "output": {"path": str(out_path), "width": w, "height": h, "fps": cfg.fps, "nb_frames": n_frames,
                   "duration_s": n_frames / cfg.fps},
        "cage_width_mm": cage_mm,
        "px_per_mm": px_per_mm,
        "cohort": entry.cohort,
        "lighting": entry.lighting,
        "notes": entry.notes,
        "params_hash": digest,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if extra_meta:
        meta.update(extra_meta)
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("%s: done -> %s (%dx%d, %d frames, %.1fs)", name, out_path.name, w, h, n_frames, meta["elapsed_s"])
    return out_path


def resident_clip_for(entry: VideoEntry, cfg: PipelineConfig, overwrite: bool = False) -> Path | None:
    """Write the short clip around intruder entry used by the resident stage; None if disabled."""
    rc = cfg.get("video.resident_clip") or {}
    if not rc.get("enabled", False):
        return None
    pre = float(rc.get("pre_roll_s", 3.0))
    start = max(entry.intruder_entry_s - pre, 0.0)
    length = (entry.intruder_entry_s - start) + pre
    entry_frame = int(round((entry.intruder_entry_s - start) * cfg.fps))
    out_dir = cfg.path("paths.resident_clips", "data/resident_clips")
    return preprocess_video(entry, cfg, out_dir=out_dir, overwrite=overwrite, start_s=start, clip_len_s=length,
                            frame_counter=False,
                            extra_meta={"resident_clip": True, "pre_roll_s": pre, "entry_frame": entry_frame,
                                        "main_clip": f"{entry.video_name}.mp4"})


def preprocess_all(entries: list[VideoEntry], cfg: PipelineConfig, overwrite: bool = False, resident_clips: bool = True) -> list[Path]:
    outputs = []
    for entry in entries:
        outputs.append(preprocess_video(entry, cfg, overwrite=overwrite))
        if resident_clips:
            resident_clip_for(entry, cfg, overwrite=overwrite)
    return outputs


def load_sidecar(cfg: PipelineConfig, video_name: str) -> dict[str, Any]:
    path = cfg.path("paths.processed_videos") / f"{video_name}.json"
    if not path.exists():
        raise PreprocessError(f"no preprocessing sidecar for {video_name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

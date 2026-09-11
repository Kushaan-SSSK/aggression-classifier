# Kushaan Sharma
"""Preprocess stage: colour path, resident clip and stale-output guard. Needs ffmpeg and OpenCV."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from behavior_pipeline.config import load_config
from behavior_pipeline.video.manifest import VideoEntry
from behavior_pipeline.video.preprocess import PreprocessError, ffmpeg_path, preprocess_all, preprocess_video, probe_video

ROOT = Path(__file__).resolve().parent.parent

try:
    ffmpeg_path(None)
    HAVE_FFMPEG = True
except PreprocessError:
    HAVE_FFMPEG = False

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def _colour_video(path: Path, seconds: float = 8.0, fps: int = 30, size=(160, 120)) -> None:
    # Orange-ish background with one dark moving dot.
    import cv2

    w, h = size
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(int(seconds * fps)):
        fr = np.full((h, w, 3), (40, 120, 200), np.uint8)
        cv2.circle(fr, (30 + t % 60, 60), 8, (10, 10, 10), -1)
        wr.write(fr)
    wr.release()


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(ROOT / "config" / "pipeline.yaml")
    c = copy.deepcopy(c)
    c.root = tmp_path
    c.set("paths.processed_videos", "proc")
    c.set("paths.resident_clips", "resident")
    c.set("video.clip_length_s", 3)
    c.set("video.resident_clip", {"enabled": True, "pre_roll_s": 1})
    return c


def _entry(tmp_path: Path) -> VideoEntry:
    src = tmp_path / "raw.mp4"
    _colour_video(src)
    return VideoEntry("vid", src, intruder_entry_s=4.0, crop=(10, 5, 120, 100), downsample_width=None, cage_width_mm=200)


def test_grey_default_and_resident_clip(cfg, tmp_path):
    entry = _entry(tmp_path)
    out = preprocess_all([entry], cfg)[0]
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["grayscale"] is True and meta["params_hash"]
    assert meta["output"]["nb_frames"] == 3 * 15 and meta["output"]["width"] == 120
    clip_meta = json.loads((tmp_path / "resident" / "vid.json").read_text())
    assert clip_meta["resident_clip"] is True and clip_meta["entry_frame"] == 15
    assert abs(clip_meta["output"]["nb_frames"] - 30) <= 1
    info = probe_video(tmp_path / "resident" / "vid.mp4")
    assert abs(info["nb_frames"] - 30) <= 1


def test_colour_path_keeps_channels_and_counter(cfg, tmp_path):
    import cv2

    entry = _entry(tmp_path)
    cfg.set("video.grayscale", False)
    out = preprocess_video(entry, cfg)
    cap = cv2.VideoCapture(str(out))
    ok, fr = cap.read()
    cap.release()
    assert ok and fr.ndim == 3
    assert np.abs(fr[:, :, 0].astype(int) - fr[:, :, 2].astype(int)).mean() > 20
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["grayscale"] is False and meta["clahe"]["enabled"] is True and meta["frame_counter"]["enabled"] is True


def test_stale_output_guard(cfg, tmp_path):
    entry = _entry(tmp_path)
    out = preprocess_video(entry, cfg)
    preprocess_video(entry, cfg)
    cfg.set("video.clahe.enabled", False)
    with pytest.raises(PreprocessError, match="different settings"):
        preprocess_video(entry, cfg)
    preprocess_video(entry, cfg, overwrite=True)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["clahe"]["enabled"] is False
    meta["adopted_from_simba_batch"] = True
    out.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(PreprocessError, match="adopted=True"):
        preprocess_video(entry, cfg)

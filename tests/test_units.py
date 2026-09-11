# Kushaan Sharma
"""Unit tests that need no SimBA project. Run: pytest tests/test_units.py"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavior_pipeline.video.adopt import load_batch_log, write_manifest_from_log
from behavior_pipeline.config import SIMBA_16BP_PER_ANIMAL, load_config
from behavior_pipeline.pose.blob import counter_rect, detect_boxes, fill_missing_boxes, predictions_to_h5
from behavior_pipeline.pose.convert import find_pose_file, link_identities, to_simba_dataframe
from behavior_pipeline.simba.proximity import bouts_from_flags, min_bout_filter
from behavior_pipeline.pose.resident import decide_swap, resident_track

ROOT = Path(__file__).resolve().parent.parent


def _two_tracks(n: int = 50, swap_at: int = 20) -> np.ndarray:
    # Two animals walking towards each other; identities swapped from swap_at onwards.
    data = np.zeros((n, 2, len(SIMBA_16BP_PER_ANIMAL), 3))
    data[..., 2] = 0.95
    for t in range(n):
        a = np.array([10 + t, 100.0])
        b = np.array([300 - t, 100.0])
        for k in range(len(SIMBA_16BP_PER_ANIMAL)):
            data[t, 0, k, :2] = a
            data[t, 1, k, :2] = b
        if t >= swap_at:
            data[t] = data[t][::-1]
    return data


def test_link_identities_undoes_swaps():
    data = _two_tracks()
    linked, stats = link_identities(data, anchor_bp="Center", max_jump_px=50)
    assert stats["n_reassigned"] == 30
    assert stats["n_jumps"] == 0
    x1 = linked[:, 0, SIMBA_16BP_PER_ANIMAL.index("Center"), 0]
    assert np.all(np.diff(x1) == 1)


def test_link_identities_handles_missing_frames():
    data = _two_tracks(swap_at=999)
    data[10:15, :, :, 0] = np.nan
    data[10:15, :, :, 2] = 0.0
    linked, stats = link_identities(data)
    assert stats["n_frames_no_anchor"] == 5
    assert linked.shape == data.shape


def test_to_simba_dataframe_layout():
    df = to_simba_dataframe(_two_tracks(n=3, swap_at=999))
    assert df.columns.names == ["scorer", "bodyparts", "coords"]
    assert df.shape == (3, 48)
    assert list(df.columns.get_level_values(2)[:3]) == ["x", "y", "likelihood"]


def test_min_bout_filter_and_bouts():
    flags = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    out = min_bout_filter(flags, 3)
    assert out.tolist() == [False, False, False, False, True, True, True, True, False, False]
    bouts = bouts_from_flags(out, fps=10)
    assert len(bouts) == 1 and bouts.iloc[0]["start_frame"] == 4 and bouts.iloc[0]["end_frame"] == 8
    assert isinstance(bouts, pd.DataFrame)


def test_predictions_to_h5_two_animals_and_padded_rows():
    bps = [f"bp{i}" for i in range(27)]
    good = np.stack([np.full(27, 10.0), np.full(27, 20.0), np.full(27, 0.9)], axis=1)
    second = good.copy()
    second[:, 0] = 100.0
    padded = np.full((27, 3), -1.0)  # DLC pads missing individuals with -1
    preds = [{"bodyparts": np.stack([good, second])}, {"bodyparts": np.stack([good, padded])}, {"bodyparts": np.zeros((0, 27, 3))}]
    df = predictions_to_h5(preds, bps, "test", 2, None)
    assert df.shape == (3, 2 * 27 * 3)
    a1x = df.xs(("animal1", "x"), level=("individuals", "coords"), axis=1).to_numpy()
    a1l = df.xs(("animal1", "likelihood"), level=("individuals", "coords"), axis=1).to_numpy()
    assert a1x[0, 0] == 100.0 and a1l[0, 0] == 0.9
    assert np.isnan(a1x[1, 0]) and a1l[1, 0] == 0.0
    assert np.isnan(df.iloc[2].xs("x", level="coords")).all()


def test_fill_missing_boxes_reuses_previous_or_full_frame():
    boxes = [[], [(1, 2, 3, 4)], [], [], [(5, 6, 7, 8), (9, 9, 2, 2)], []]
    filled, n = fill_missing_boxes(boxes, (100, 50))
    assert n == 4
    assert filled[0] == [(0, 0, 50, 100)]
    assert filled[2] == [(1, 2, 3, 4)] and filled[3] == [(1, 2, 3, 4)]
    assert filled[5] == [(5, 6, 7, 8), (9, 9, 2, 2)]
    assert all(len(b) >= 1 for b in filled)


def test_detect_boxes_ignores_counter_rect_and_oversize_blobs():
    from behavior_pipeline.pose.blob import DEFAULTS

    H, W = 300, 200
    bg = np.full((H, W), 200, np.uint8)
    frame = bg.copy()
    frame[150:180, 60:90] = 20  # mouse-sized blob
    frame[5:25, 5:80] = 20  # frame-counter text, top left
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    rect = counter_rect((H, W), cfg)
    assert rect is not None and rect[0] <= 5 and rect[1] <= 5 and rect[2] >= 80 and rect[3] >= 25
    params = dict(DEFAULTS)
    boxes, largest, dropped = detect_boxes(frame, bg, params, None, ignore_rect=rect)
    assert len(boxes) == 1 and dropped == 0
    x, y, w, h = boxes[0]
    assert 40 <= x <= 60 and 130 <= y <= 150
    boxes_no_mask, _, _ = detect_boxes(frame, bg, params, None, ignore_rect=None)
    assert len(boxes_no_mask) == 2
    frame[100:280, 20:180] = 20  # blob covering half the frame
    boxes, largest, dropped = detect_boxes(frame, bg, params, None, ignore_rect=rect)
    assert dropped == 1 and largest is None and boxes == []


def test_carry_box_keeps_the_other_mouse_but_not_a_merged_pair():
    from behavior_pipeline.pose.blob import _carry_box, _iou

    a, b = (10, 10, 40, 40), (200, 200, 40, 40)
    assert _iou(a, a) == 1.0 and _iou(a, b) == 0.0
    assert _carry_box((12, 12, 40, 40), [a, b]) == b
    assert _carry_box((198, 202, 40, 40), [a, b]) == a
    merged = (10, 10, 230, 230)  # one blob covering both previous boxes
    assert _carry_box(merged, [a, b]) is None


def test_split_box_geometric_follows_previous_pair_axis():
    from behavior_pipeline.pose.blob import _split_box_geometric

    merged = (100, 100, 80, 200)
    halves = _split_box_geometric(merged)
    assert halves == [(100, 100, 80, 120), (100, 180, 80, 120)]
    halves = _split_box_geometric((100, 100, 200, 80), [(100, 100, 60, 80), (240, 100, 60, 80)])
    assert halves == [(100, 100, 120, 80), (180, 100, 120, 80)]
    halves = _split_box_geometric(merged, [(100, 100, 80, 90), (100, 210, 80, 90)])
    assert halves[0][1] == 100 and halves[1][1] == 180


def test_detect_boxes_splits_a_long_component():
    from behavior_pipeline.pose.blob import DEFAULTS

    H, W = 700, 400
    bg = np.full((H, W), 200, np.uint8)
    frame = bg.copy()
    frame[200:440, 170:230] = 20  # two mice in a line, 60 x 240 px
    boxes, largest, dropped = detect_boxes(frame, bg, dict(DEFAULTS), typical_area=9000.0)
    assert len(boxes) == 2 and dropped == 0
    ys = sorted(y + h / 2 for _, y, _, h in boxes)
    assert ys[0] < 320 < ys[1]
    single = bg.copy()
    single[300:420, 170:250] = 20  # one mouse, 80 x 120 px
    boxes, _, _ = detect_boxes(single, bg, dict(DEFAULTS), typical_area=9000.0)
    assert len(boxes) == 1


def test_resident_track_follows_single_pre_entry_blob():
    boxes = [[(10, 10, 20, 20)]] * 5 + [[]] * 2 + [[(14, 14, 20, 20), (100, 100, 20, 20)]] * 5
    track, frac = resident_track(boxes, entry_frame=6, max_jump_px=30)
    assert frac == 5 / 6
    assert np.allclose(track[0], [20, 20])
    assert np.allclose(track[5], [20, 20]) and np.allclose(track[6], [20, 20])
    assert np.allclose(track[-1], [24, 24])


def test_decide_swap_cases():
    n = 30
    res = np.tile([50.0, 50.0], (n, 1))
    near = res + 8.0
    far = np.tile([300.0, 300.0], (n, 1))
    assert decide_swap(res, near, far)["swap"] is False
    assert decide_swap(res, far, near)["swap"] is True
    assert decide_swap(res, far, far + 5)["swap"] is None
    assert decide_swap(res, near, near + 1)["swap"] is None
    assert decide_swap(res[:5], near[:5], far[:5])["swap"] is None
    res_nan = res.copy()
    res_nan[:25] = np.nan
    assert decide_swap(res_nan, near, far)["swap"] is None


def test_write_manifest_from_log_source_dir_and_overwrite(tmp_path):
    log = {"video_data": {
        "1.2.26.ELS1": {"video_info": {"file_path": "C:/lab/Box/Videos\\1.2.26.ELS1.mp4", "width": 1920, "height": 1080, "fps": 60.0},
                         "crop": True, "crop_settings": {"top_left_x": 10, "top_left_y": 20, "width": 300, "height": 700},
                         "clip": True, "clip_settings": {"start": "00:00:07", "stop": "00:05:07"}},
        "1.2.26.ELS2_do_not_use": {"video_info": {"file_path": "C:/lab/Box/Videos\\1.2.26.ELS2_do_not_use.mp4", "width": 1920, "height": 720, "fps": 30.0},
                                   "crop": False, "clip": False},
    }}
    log_path = tmp_path / "batch_process_log.json"
    log_path.write_text(json.dumps(log), encoding="utf-8")
    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    out = tmp_path / "manifest.csv"
    write_manifest_from_log(load_batch_log(log_path), cfg, out, source_dir=tmp_path / "orig", exclude=("do_not_use",))
    df = pd.read_csv(out)
    assert list(df["video_name"]) == ["1-2-26_ELS1"]
    assert df.loc[0, "source_path"] == str(tmp_path / "orig" / "1.2.26.ELS1.mp4")
    assert df.loc[0, "intruder_entry_s"] == 7 and df.loc[0, "crop_w"] == 300
    assert "1920x1080" in df.loc[0, "notes"]
    with pytest.raises(FileExistsError):
        write_manifest_from_log(load_batch_log(log_path), cfg, out)
    write_manifest_from_log(load_batch_log(log_path), cfg, out, overwrite=True)
    assert len(pd.read_csv(out)) == 2 and list(tmp_path.glob("manifest.*.bak.csv"))


def test_find_pose_file_prefers_newest_and_exact_prefix(tmp_path):
    old = tmp_path / "vid1_blobbox_a.h5"
    old.write_bytes(b"0")
    other = tmp_path / "vid10_blobbox_a.h5"  # different video whose name starts with vid1
    other.write_bytes(b"0")
    time.sleep(0.05)
    new = tmp_path / "vid1_superanimal_b.h5"
    new.write_bytes(b"0")
    now = time.time()
    import os

    os.utime(old, (now - 100, now - 100))
    os.utime(other, (now + 100, now + 100))
    assert find_pose_file("vid1", tmp_path) == new
    assert find_pose_file("vid10", tmp_path) == other

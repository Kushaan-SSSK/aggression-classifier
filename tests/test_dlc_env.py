# Kushaan Sharma
"""DeepLabCut-dependent checks. Run: .venv-dlc\\Scripts\\python -m pytest tests\\test_dlc_env.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("deeplabcut")

from behavior_pipeline.config import load_config
from behavior_pipeline.pose.blob import build_pose_config, is_top_down, run_pose_on_boxes

ROOT = Path(__file__).resolve().parent.parent


def test_blob_config_is_top_down():
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig

    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    config, sa_name, model_name, max_ind = build_pose_config(cfg, "cpu")
    assert is_top_down(config)
    assert max_ind == 2 and len(config.metadata.bodyparts) == 27
    bu = PoseConfig.build_for_superanimal_inference(sa_name, model_name=model_name, detector_name=None, max_individuals=2, device="cpu")
    assert not is_top_down(bu)


def test_two_boxes_yield_two_animals(tmp_path):
    import cv2

    cfg = load_config(ROOT / "config" / "pipeline.yaml")
    video = tmp_path / "two.mp4"
    # Two dark ellipses on a light background, one per box.
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (200, 200))
    for _ in range(3):
        fr = np.full((200, 200, 3), 180, np.uint8)
        cv2.ellipse(fr, (50, 50), (25, 12), 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(fr, (150, 150), (25, 12), 90, 0, 360, (20, 20, 20), -1)
        wr.write(fr)
    wr.release()
    boxes = [[(15, 25, 70, 50), (125, 115, 50, 70)]] * 3
    preds, bodyparts, scorer = run_pose_on_boxes(video, boxes, cfg, device="cpu")
    assert len(preds) == 3 and len(bodyparts) == 27 and scorer.startswith("blobbox_")
    arr = np.asarray(preds[0]["bodyparts"])
    assert arr.shape[0] == 2
    assert (arr[0, :, 2] > 0).any() and (arr[1, :, 2] > 0).any()
    assert arr[0, :, 0].mean() < 100 < arr[1, :, 0].mean()

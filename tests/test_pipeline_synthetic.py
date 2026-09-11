# Kushaan Sharma
"""End-to-end test on synthetic data. Run inside .venv-simba: pytest tests/test_pipeline_synthetic.py
The tests are ordered and share one project; each stage feeds the next.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavior_pipeline.config import SIMBA_16BP_COLUMNS, load_config

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config" / "pipeline.yaml"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    from tests.synthetic import VIDEO_NAME, make_synthetic_project

    root = tmp_path_factory.mktemp("synth")
    cfg_path = make_synthetic_project(root, TEMPLATE, seconds=45.0)
    cfg = load_config(cfg_path)
    return {"cfg": cfg, "root": root, "video": VIDEO_NAME}


def test_01_manifest_and_annotations(project):
    from behavior_pipeline.video.manifest import read_annotations, read_video_manifest

    cfg = project["cfg"]
    entries = read_video_manifest(cfg.path("paths.video_manifest"), cfg)
    assert len(entries) == 1 and entries[0].crop == (40, 20, 560, 440)
    assert entries[0].source_path.exists()
    rows = read_annotations(cfg.path("paths.annotations"), cfg)
    assert {r.behavior for r in rows} <= {"Attack", "Social_investigation"}
    assert len(rows) > 0


def test_02_preprocess(project):
    from behavior_pipeline.video.manifest import read_video_manifest
    from behavior_pipeline.video.preprocess import preprocess_all, probe_video

    cfg = project["cfg"]
    out = preprocess_all(read_video_manifest(cfg.path("paths.video_manifest"), cfg), cfg)
    assert len(out) == 1 and out[0].exists()
    meta = json.loads(out[0].with_suffix(".json").read_text())
    info = probe_video(out[0], cfg)
    assert info["width"] == 560 and info["height"] == 440
    assert abs(info["fps"] - 15) < 0.01
    expected = int((45 - 5) * 15)
    assert abs(meta["output"]["nb_frames"] - expected) <= 2
    assert meta["px_per_mm"] == pytest.approx(560 / 290, rel=1e-3)


def test_03_convert_pose(project):
    from behavior_pipeline.pose.convert import convert_all

    cfg = project["cfg"]
    outs = convert_all(cfg)
    assert len(outs) == 1
    df = pd.read_csv(outs[0], header=[0, 1, 2], index_col=0)
    assert list(df.columns.get_level_values(1).unique()) == SIMBA_16BP_COLUMNS
    assert df.shape[1] == 48
    qc = json.loads(outs[0].with_name(f"{project['video']}.qc.json").read_text())
    assert qc["identity_linking"]["n_reassigned"] > 0
    # After linking, Animal_1 should follow the true resident trajectory.
    truth = np.load(project["root"] / "data" / "raw_videos" / f"{project['video']}_truth.npy")
    from tests.synthetic import processed_frame_positions

    proc = processed_frame_positions(truth, 15.0, 300.0)
    n = min(len(df), len(proc))
    c1 = df.xs("Center_1", level=1, axis=1).iloc[:n]
    err = np.hypot(c1.iloc[:, 0].to_numpy() - proc[:n, 0, 0], c1.iloc[:, 1].to_numpy() - proc[:n, 0, 1])
    assert np.nanmedian(err) < 10


def test_04_simba_project(project):
    from behavior_pipeline.simba.project import build_project

    cfg = project["cfg"]
    feats = build_project(cfg)
    files = list(feats.glob("*.csv"))
    assert len(files) == 1
    df = pd.read_csv(files[0], index_col=0)
    assert len(df) > 500 and df.shape[1] > 100
    info = pd.read_csv(cfg.simba_project_folder / "logs" / "video_info.csv")
    assert list(info.columns) == ["Video", "fps", "Resolution_width", "Resolution_height", "Distance_in_mm", "pixels/mm"]


def test_05_annotate(project):
    from behavior_pipeline.simba.annotations import build_targets

    cfg = project["cfg"]
    written = build_targets(cfg)
    assert len(written) == 1
    df = pd.read_csv(written[0], index_col=0)
    assert df["Attack"].sum() > 0 and df["Social_investigation"].sum() > 0
    assert set(df["Attack"].unique()) <= {0, 1}


def test_06_train(project):
    from behavior_pipeline.simba.train import train_all

    cfg = project["cfg"]
    models = train_all(cfg, n_estimators=30)
    assert set(models) == {"Attack", "Social_investigation"}
    assert all(p.exists() for p in models.values())


def test_07_infer(project):
    from behavior_pipeline.simba.infer import infer_all

    cfg = project["cfg"]
    out = infer_all(cfg)
    df = pd.read_csv(next(out.glob("*.csv")), index_col=0)
    assert "Probability_Attack" in df.columns and "Attack" in df.columns
    assert df["Probability_Attack"].between(0, 1).all()


def test_08_proximity(project):
    from behavior_pipeline.simba.proximity import compute_proximity

    cfg = project["cfg"]
    table = compute_proximity(cfg)
    assert len(table) == 1 and table.iloc[0]["n_bouts"] >= 1
    mr = pd.read_csv(cfg.simba_project_folder / "csv" / "machine_results" / f"{project['video']}.csv", index_col=0)
    assert "Proximity" in mr.columns


def test_09_unsupervised(project):
    import subprocess
    import sys

    # Run in a fresh interpreter, as the CLI does; umap is very slow inside the pytest process.
    cfg = project["cfg"]
    cmd = [sys.executable, "-m", "behavior_pipeline", "unsupervised", "--config", str(cfg.config_path)]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    out = cfg.simba_project_folder / "unsupervised"
    assert list((out / "umap").glob("*.pickle"))
    assert list((out / "hdbscan").glob("*.pickle"))

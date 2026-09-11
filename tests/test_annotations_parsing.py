# Kushaan Sharma
"""Annotation CSV parsing: video name spellings, time formats, raw-video time reference."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from behavior_pipeline.config import load_config
from behavior_pipeline.video.manifest import ManifestError, parse_time_s, read_annotations, resolve_video_name, safe_video_name, video_name_lookup

ROOT = Path(__file__).resolve().parent.parent


def test_parse_time_formats():
    assert parse_time_s("95") == 95.0
    assert parse_time_s("95.5") == 95.5
    assert parse_time_s("1:35") == 95.0
    assert parse_time_s("1:35.5") == 95.5
    assert parse_time_s("0:01:35") == 95.0
    assert parse_time_s("") is None
    with pytest.raises(ManifestError):
        parse_time_s("1:2:3:4", "start_s", 3)
    with pytest.raises(ManifestError):
        parse_time_s("abc", "start_s", 3)


def test_safe_video_name_variants():
    assert safe_video_name("10.3.25.ELS580") == "10-3-25_ELS580"
    assert safe_video_name("10.3.25.ELS580.mp4") == "10-3-25_ELS580"
    assert safe_video_name("10-3-25_ELS580") == "10-3-25_ELS580"
    assert safe_video_name("ELS 580 (a)") == "ELS_580_a_"


@pytest.fixture()
def cfg(tmp_path):
    # One processed video with the intruder entering at 5 s.
    c = copy.deepcopy(load_config(ROOT / "config" / "pipeline.yaml"))
    c.root = tmp_path
    c.set("paths.processed_videos", "proc")
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "10-3-25_ELS580.json").write_text(json.dumps({
        "video_name": "10-3-25_ELS580", "source_path": "C:/x/Preprocessed Videos/10.3.25.ELS580.mp4", "intruder_entry_s": 5.0,
        "output": {"fps": 15, "duration_s": 31.8}}))
    return c


def test_video_name_lookup_and_resolution(cfg):
    lookup = video_name_lookup(cfg)
    for spelling in ("10-3-25_ELS580", "10.3.25.ELS580", "10.3.25.ELS580.mp4", "10.3.25.els580"):
        assert resolve_video_name(spelling, lookup) == "10-3-25_ELS580"
    assert resolve_video_name("8.1.26.ELS999", lookup) == "8-1-26_ELS999"


def test_read_annotations_tolerant(cfg, tmp_path):
    csv = tmp_path / "ann.csv"
    csv.write_text(
        "Video,Behaviour,Start,End,annotator\n"
        "10.3.25.ELS580,attack,0:10,0:12.5,NR\n"
        "10.3.25.ELS580,Social investigation,20,25,NR\n"
        "10.3.25.ELS580,fighting,1:00,1:05,NR\n"
    )
    rows = read_annotations(csv, cfg)
    assert [r.video_name for r in rows] == ["10-3-25_ELS580"] * 3
    assert [r.behavior for r in rows] == ["Attack", "Social_investigation", "Attack"]
    assert (rows[0].start_s, rows[0].end_s) == (10.0, 12.5)


def test_read_annotations_raw_time_reference(cfg, tmp_path):
    csv = tmp_path / "ann.csv"
    # Raw times are shifted by the 5 s entry: 7-12 -> 2-7, 1-3 dropped, 4-6 clipped to 0-1.
    csv.write_text(
        "video_name,behavior,start_s,end_s,time_reference\n"
        "10.3.25.ELS580,attack,0:07,0:12,raw\n"
        "10.3.25.ELS580,attack,0:01,0:03,raw\n"
        "10.3.25.ELS580,attack,0:04,0:06,raw\n"
        "10.3.25.ELS580,sniff,3,4,\n"
    )
    rows = read_annotations(csv, cfg)
    assert [(r.start_s, r.end_s) for r in rows] == [(2.0, 7.0), (0.0, 1.0), (3.0, 4.0)]
    cfg.set("annotations.time_reference", "raw")
    csv.write_text("video_name,behavior,start_s,end_s\n10.3.25.ELS580,attack,10,12\n")
    rows = read_annotations(csv, cfg)
    assert (rows[0].start_s, rows[0].end_s) == (5.0, 7.0)


def test_read_annotations_rejects_todo(cfg, tmp_path):
    csv = tmp_path / "ann.csv"
    csv.write_text("video_name,behavior,start_s,end_s\n10-3-25_ELS580,TODO,,\n")
    with pytest.raises(ManifestError, match="unknown behavior"):
        read_annotations(csv, cfg)

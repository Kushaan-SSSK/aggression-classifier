# Kushaan Sharma

"""Pose benchmark: runs a short clip through several preprocessing and detection variants and scores each.

Run in .venv-dlc:  python -m behavior_pipeline pose --bench --videos <name> [<name> ...]
Writes <paths.pose_bench>/bench_results.csv and .json.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import LOG, SIMBA_16BP_PER_ANIMAL, PipelineConfig
from ..video.manifest import VideoEntry, read_video_manifest
from ..video.preprocess import preprocess_video, probe_video

VARIANTS: dict[str, dict[str, Any]] = {
    "grey_clahe_blob": {"grayscale": True, "clahe": True, "mode": "blob"},
    "grey_blob": {"grayscale": True, "clahe": False, "mode": "blob"},
    "colour_blob": {"grayscale": False, "clahe": False, "mode": "blob"},
    "grey_clahe_det06": {"grayscale": True, "clahe": True, "mode": "detector", "box_score_thresh": 0.6},
    "colour_det06": {"grayscale": False, "clahe": False, "mode": "detector", "box_score_thresh": 0.6},
    "colour_det03": {"grayscale": False, "clahe": False, "mode": "detector", "box_score_thresh": 0.3},
}


def bench_metrics(h5: Path, cfg: PipelineConfig, n_video_frames: int) -> dict[str, Any]:
    """Track quality of one pose file, read and identity-linked as the convert stage does."""
    from .convert import link_identities, read_pose

    kp_map = cfg.get("pose.keypoint_map.superanimal")
    ident = cfg.get("pose.identity") or {}
    min_lik = float(ident.get("min_likelihood", 0.1))
    data, _ = read_pose(h5, kp_map, n_animals=2)
    linked, stats = link_identities(data, anchor_bp=str(ident.get("anchor_body_part", "Center")),
                                    max_jump_px=float(ident.get("max_jump_px", 150)), min_likelihood=min_lik)
    c = SIMBA_16BP_PER_ANIMAL.index("Center")
    ok = (~np.isnan(linked[:, :, c, 0])) & (linked[:, :, c, 2] >= min_lik)
    lik = linked[..., 2]
    with np.errstate(invalid="ignore"):
        mean_lik = [float(np.nanmean(np.where(lik[:, a] > 0, lik[:, a], np.nan))) if (lik[:, a] > 0).any() else 0.0 for a in range(2)]
    return {
        "n_frames_h5": int(len(linked)), "n_frames_video": int(n_video_frames), "frames_match": bool(len(linked) == n_video_frames),
        "pct_frames_2_animals": round(float(ok.all(axis=1).mean() * 100), 1),
        "pct_frames_1_animal": round(float((ok.sum(axis=1) == 1).mean() * 100), 1),
        "mean_lik_8kp_animal1": round(mean_lik[0], 3), "mean_lik_8kp_animal2": round(mean_lik[1], 3),
        "mean_lik_8kp": round(float(np.mean(mean_lik)), 3),
        "n_jumps": int(stats.get("n_jumps", 0)), "n_reassigned": int(stats.get("n_reassigned", 0)),
    }


def _variant_cfg(cfg: PipelineConfig, spec: dict[str, Any]) -> PipelineConfig:
    v = copy.deepcopy(cfg)
    v.set("video.grayscale", bool(spec["grayscale"]))
    v.set("video.clahe.enabled", bool(spec["clahe"]))
    v.set("pose.superanimal.create_labeled_video", False)
    if spec.get("box_score_thresh") is not None:
        v.set("pose.superanimal.box_score_thresh", float(spec["box_score_thresh"]))
    return v


def run_bench(cfg: PipelineConfig, video_names: list[str], segment_start_s: float = 10.0, segment_len_s: float = 20.0,
              variants: list[str] | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Run every variant on a clip of each named manifest video and tabulate the metrics."""
    from .blob import run_blob_pose
    from .superanimal import resolve_device, run_superanimal

    out_dir = out_dir or cfg.path("paths.pose_bench", "data/pose/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {e.video_name: e for e in read_video_manifest(cfg.path("paths.video_manifest"), cfg)}
    missing = [n for n in video_names if n not in entries]
    if missing:
        raise KeyError(f"not in the video manifest: {missing}")
    names = variants or list(VARIANTS)
    device = resolve_device(cfg.get("pose.superanimal.device", "auto"))
    rows: list[dict[str, Any]] = []
    prev_csv = out_dir / "bench_results.csv"
    if prev_csv.exists():
        # keep earlier rows for (video, variant) pairs that are not rerun now
        prev = pd.read_csv(prev_csv)
        rows = [r for r in prev.to_dict("records") if not (r["video"] in set(video_names) and r["variant"] in set(names))]
    for name in video_names:
        entry: VideoEntry = entries[name]
        vdir = out_dir / name
        vdir.mkdir(parents=True, exist_ok=True)
        for var in names:
            spec = VARIANTS[var]
            vcfg = _variant_cfg(cfg, spec)
            clip = preprocess_video(entry, vcfg, out_dir=vdir, overwrite=True, start_s=entry.intruder_entry_s + segment_start_s,
                                    clip_len_s=segment_len_s, name=f"{name}__{var}")
            n_video = probe_video(clip, vcfg)["nb_frames"]
            t0 = time.time()
            try:
                if spec["mode"] == "blob":
                    h5 = run_blob_pose([clip], vcfg, dest=vdir, preview=True, device=device)[0]
                else:
                    h5 = run_superanimal([clip], vcfg, dest=vdir, box_score_thresh=spec.get("box_score_thresh"))[0]
                elapsed = time.time() - t0
                m = bench_metrics(h5, vcfg, n_video)
                blobstats = vdir / f"{clip.stem}_blobbox_{vcfg.get('pose.superanimal.superanimal_name')}_{vcfg.get('pose.superanimal.model_name')}.blobstats.json"
                if blobstats.exists():
                    bs = json.loads(blobstats.read_text(encoding="utf-8"))
                    m["pct_frames_2_boxes"] = round(100.0 * bs["frames_2_boxes"] / max(bs["n_frames"], 1), 1)
                row = {"video": name, "variant": var, **spec, "sec_per_frame": round(elapsed / max(n_video, 1), 3), "h5": str(h5), **m}
            except Exception as exc:
                LOG.error("%s / %s failed: %s", name, var, exc)
                row = {"video": name, "variant": var, **spec, "error": str(exc)[:300]}
            rows.append(row)
            LOG.info("bench %s / %s: %s", name, var, {k: v for k, v in row.items() if k in ("pct_frames_2_animals", "mean_lik_8kp", "sec_per_frame", "n_jumps", "error")})
            pd.DataFrame(rows).to_csv(out_dir / "bench_results.csv", index=False)
            (out_dir / "bench_results.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    table = pd.DataFrame(rows)
    show = [c for c in ("video", "variant", "pct_frames_2_animals", "pct_frames_1_animal", "mean_lik_8kp", "n_jumps", "sec_per_frame", "frames_match", "pct_frames_2_boxes", "error") if c in table.columns]
    print(table[show].to_string(index=False))
    return table

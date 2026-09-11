# Kushaan Sharma

"""Zero-shot pose with the DeepLabCut 3 SuperAnimal-TopViewMouse detector and pose head.

Run in .venv-dlc:  python -m behavior_pipeline pose --mode detector [--videos ...]
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import yaml

from ..config import LOG, PipelineConfig

FAST_PROFILE = {"detector_name": "fasterrcnn_mobilenet_v3_large_fpn"}


def resolve_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def write_custom_config(cfg: PipelineConfig, dest: Path, device: str, box_score_thresh: float, detector_name: str | None = None) -> Path:
    """Dump the SuperAnimal inference config with a different detector box_score_thresh."""
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig

    sa = cfg.get("pose.superanimal") or {}
    sa_name = sa.get("superanimal_name", "superanimal_topviewmouse")
    model_name = sa.get("model_name", "hrnet_w32")
    det = detector_name or sa.get("detector_name", "fasterrcnn_resnet50_fpn_v2")
    config = PoseConfig.build_for_superanimal_inference(sa_name, model_name=model_name, detector_name=det,
                                                        max_individuals=int(sa.get("max_individuals", 2)), device=device)
    d = _plain(config.to_dict())
    d.setdefault("detector", {}).setdefault("model", {})["box_score_thresh"] = float(box_score_thresh)
    path = dest / f"_{sa_name}_{model_name}_{det}_thr{box_score_thresh:g}.yaml"
    path.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return path


def _plain(obj):
    """Recursively turn enums, paths and numpy scalars into YAML-safe values."""
    from enum import Enum

    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Enum):
        return _plain(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def run_superanimal(videos: list[str | Path], cfg: PipelineConfig, dest: Path | None = None, fast: bool = False,
                    box_score_thresh: float | None = None) -> list[Path]:
    """Run deeplabcut.video_inference_superanimal on the videos; fast swaps in the MobileNet detector."""
    try:
        import deeplabcut
    except ImportError as exc:
        raise RuntimeError("DeepLabCut is not installed in this interpreter; use .venv-dlc") from exc

    sa = dict(cfg.get("pose.superanimal") or {})
    if fast:
        sa.update(FAST_PROFILE)
    dest = dest or cfg.path("paths.pose_raw")
    dest.mkdir(parents=True, exist_ok=True)
    device = resolve_device(sa.get("device", "auto"))
    video_paths = [str(Path(v).resolve()) for v in videos]
    if not video_paths:
        raise RuntimeError("no videos given")
    thr = box_score_thresh if box_score_thresh is not None else sa.get("box_score_thresh")
    LOG.info("SuperAnimal inference on %d video(s), device=%s, max_individuals=%s, box_score_thresh=%s",
             len(video_paths), device, sa.get("max_individuals", 2), thr if thr is not None else "default")
    kwargs = dict(
        videos=video_paths,
        superanimal_name=sa.get("superanimal_name", "superanimal_topviewmouse"),
        model_name=sa.get("model_name", "hrnet_w32"),
        detector_name=sa.get("detector_name", "fasterrcnn_resnet50_fpn_v2"),
        max_individuals=int(sa.get("max_individuals", 2)),
        video_adapt=bool(sa.get("video_adapt", False)),
        pcutoff=float(sa.get("pcutoff", 0.1)),
        bbox_threshold=float(sa.get("bbox_threshold", 0.9)),
        batch_size=int(sa.get("batch_size", 8)),
        dest_folder=str(dest),
        device=device,
        create_labeled_video=bool(sa.get("create_labeled_video", True)),
    )
    if sa.get("scale_list"):
        kwargs["scale_list"] = list(sa["scale_list"])
    if thr is not None:
        # bbox_threshold only affects plotting; the detector's real filter lives in the model config
        kwargs["customized_model_config"] = str(write_custom_config(cfg, dest, device, float(thr), detector_name=kwargs["detector_name"]))
    deeplabcut.video_inference_superanimal(**kwargs)

    produced: list[Path] = []
    for v in video_paths:
        stem = Path(v).stem
        hits = sorted(dest.glob(f"{stem}*.h5"))
        if not hits:
            # some DLC versions write next to the video
            hits = sorted(Path(v).parent.glob(f"{stem}*.h5"))
            for h in hits:
                shutil.move(str(h), dest / h.name)
            hits = sorted(dest.glob(f"{stem}*.h5"))
        if not hits:
            LOG.warning("%s: no H5 output found in %s", stem, dest)
        produced.extend(hits)
    LOG.info("SuperAnimal finished: %d H5 file(s) in %s", len(produced), dest)
    return produced


def expand_video_args(patterns: list[str], cfg: PipelineConfig) -> list[Path]:
    """Accept explicit paths or globs; default to every mp4 in processed_videos."""
    if not patterns:
        return sorted(cfg.path("paths.processed_videos").glob("*.mp4"))
    out: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(pat)]
        out.extend(hits if hits else [Path(pat)])
    return out


def pose_status(cfg: PipelineConfig) -> list[dict[str, object]]:
    """Which processed videos have a pose H5 newer than the video."""
    from .convert import find_pose_file

    rows = []
    pose_dir = cfg.path("paths.pose_raw")
    for mp4 in sorted(cfg.path("paths.processed_videos").glob("*.mp4")):
        h5 = find_pose_file(mp4.stem, pose_dir) if pose_dir.exists() else None
        fresh = bool(h5 and h5.stat().st_mtime >= mp4.stat().st_mtime)
        rows.append({"video": mp4.stem, "pose_file": h5.name if h5 else "", "status": "ok" if fresh else ("stale" if h5 else "missing")})
    return rows

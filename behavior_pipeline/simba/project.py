# Kushaan Sharma
"""Create and populate the SimBA project: videos, video_info.csv, pose import,
outlier correction and feature extraction. Run as the `project` stage of the CLI.
"""
from __future__ import annotations

import configparser
import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import LOG, PipelineConfig

VIDEO_INFO_COLUMNS = ["Video", "fps", "Resolution_width", "Resolution_height", "Distance_in_mm", "pixels/mm"]


def _require_simba():
    try:
        import simba
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SimBA is not installed in this interpreter; use .venv-simba") from exc


def create_project(cfg: PipelineConfig, overwrite: bool = False) -> Path:
    """Create the SimBA project tree if needed and return the project_config.ini path."""
    _require_simba()
    from simba.utils.config_creator import ProjectConfigCreator

    config_path = cfg.simba_config_path
    if config_path.exists() and not overwrite:
        LOG.info("SimBA project exists: %s", config_path)
        _ensure_classifiers(cfg)
        return config_path
    if config_path.exists() and overwrite:
        shutil.rmtree(cfg.simba_project_folder.parent)
    project_path = cfg.simba_project_dir
    project_path.mkdir(parents=True, exist_ok=True)
    ProjectConfigCreator(
        project_path=str(project_path),
        project_name=str(cfg.get("project.simba_project_name")),
        target_list=list(cfg.classifiers),
        pose_estimation_bp_cnt=str(cfg.get("simba.pose_estimation_bp_cnt", "16")),
        body_part_config_idx=int(cfg.get("simba.body_part_config_idx", 6)),
        animal_cnt=int(cfg.get("simba.animal_cnt", 2)),
        file_type=str(cfg.get("project.workflow_file_type", "csv")),
    )
    if not config_path.exists():
        raise RuntimeError(f"SimBA did not create {config_path}")
    LOG.info("created SimBA project at %s", config_path)
    return config_path


def _ensure_classifiers(cfg: PipelineConfig, extra: list[str] | None = None) -> None:
    """Add any configured (or extra) classifier names missing from project_config.ini."""
    ini = _read_ini(cfg.simba_config_path)
    section = "SML settings"
    if not ini.has_section(section):
        return
    existing = [ini.get(section, k) for k in ini[section] if k.startswith("target_name_")]
    changed = False
    for clf in list(cfg.classifiers) + list(extra or []):
        if clf not in existing:
            existing.append(clf)
            changed = True
    if changed:
        for i, clf in enumerate(existing, start=1):
            ini.set(section, f"target_name_{i}", clf)
            ini.set(section, f"model_path_{i}", "")
        ini.set(section, "no_targets", str(len(existing)))
        _write_ini(ini, cfg.simba_config_path)
        LOG.info("added classifiers to project: %s", existing)


def _read_ini(path: Path) -> configparser.ConfigParser:
    ini = configparser.ConfigParser(interpolation=None)
    ini.optionxform = str
    ini.read(path, encoding="utf-8")
    return ini


def _write_ini(ini: configparser.ConfigParser, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        ini.write(fh)


def set_ini(cfg: PipelineConfig, section: str, values: dict[str, object]) -> None:
    """Write key/values into one section of project_config.ini, creating it if needed."""
    ini = _read_ini(cfg.simba_config_path)
    if not ini.has_section(section):
        ini.add_section(section)
    for k, v in values.items():
        ini.set(section, k, "" if v is None else str(v))
    _write_ini(ini, cfg.simba_config_path)


def import_videos(cfg: PipelineConfig, video_paths: list[Path] | None = None, copy: bool = True) -> list[Path]:
    dest = cfg.simba_project_folder / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    video_paths = video_paths or sorted(cfg.path("paths.processed_videos").glob("*.mp4"))
    out = []
    for vp in video_paths:
        target = dest / vp.name
        if not target.exists():
            if copy:
                shutil.copy2(vp, target)
            else:
                try:
                    target.symlink_to(vp.resolve())
                except OSError:
                    shutil.copy2(vp, target)
        out.append(target)
    LOG.info("%d video(s) in %s", len(out), dest)
    return out


def write_video_info(cfg: PipelineConfig, video_names: list[str] | None = None) -> Path:
    """Build logs/video_info.csv from the preprocessing sidecars."""
    from ..video.preprocess import probe_video

    logs = cfg.simba_project_folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    info_path = logs / "video_info.csv"
    videos_dir = cfg.simba_project_folder / "videos"
    names = video_names or sorted(p.stem for p in videos_dir.glob("*.mp4"))
    proc_dir = cfg.path("paths.processed_videos")
    cage_default = cfg.get("simba.cage_width_mm")
    px_override = cfg.get("simba.px_per_mm")
    rows = []
    for name in names:
        sidecar = proc_dir / f"{name}.json"
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            w, h, fps = meta["output"]["width"], meta["output"]["height"], meta["output"]["fps"]
            px = meta.get("px_per_mm")
            cage = meta.get("cage_width_mm") or cage_default
        else:
            vid = videos_dir / f"{name}.mp4"
            p = probe_video(vid, cfg)
            w, h, fps = p["width"], p["height"], p["fps"]
            px, cage = None, cage_default
        if px_override is not None:
            px = float(px_override)
        if px is None:
            if not cage:
                raise ValueError(f"{name}: cannot derive pixels/mm - set simba.px_per_mm or simba.cage_width_mm")
            px = round(w / float(cage), 4)
        rows.append({"Video": name, "fps": fps, "Resolution_width": w, "Resolution_height": h,
                     "Distance_in_mm": cage or 0, "pixels/mm": px})
    df = pd.DataFrame(rows, columns=VIDEO_INFO_COLUMNS)
    df.to_csv(info_path, index=False)
    LOG.info("wrote %s (%d videos)", info_path, len(df))
    return info_path


def import_pose(cfg: PipelineConfig, pose_dir: Path | None = None) -> Path:
    """Import the converted 16-bp CSVs into csv/input_csv with SimBA's DLC importer."""
    _require_simba()
    from simba.pose_importers.dlc_importer_csv import import_dlc_csv_data

    pose_dir = pose_dir or cfg.path("paths.pose_simba")
    csvs = [p for p in pose_dir.glob("*.csv")]
    if not csvs:
        raise FileNotFoundError(f"no converted pose CSVs in {pose_dir}; run the convert stage first")
    interp = cfg.get("simba.interpolation")
    smooth = cfg.get("simba.smoothing")
    interpolation_settings = {"type": str(interp["type"]), "method": str(interp["method"])} if interp else None
    smoothing_settings = {"time_window": int(smooth["time_window"]), "method": str(smooth["method"])} if smooth else None
    LOG.info("importing %d pose CSV(s) with interpolation=%s smoothing=%s", len(csvs), interpolation_settings, smoothing_settings)
    import_dlc_csv_data(config_path=str(cfg.simba_config_path), data_path=str(pose_dir),
                        interpolation_settings=interpolation_settings, smoothing_settings=smoothing_settings)
    input_dir = cfg.simba_project_folder / "csv" / "input_csv"
    n = len(list(input_dir.glob("*.csv")))
    if n == 0:
        raise RuntimeError("SimBA import produced no files in csv/input_csv")
    LOG.info("%d file(s) in %s", n, input_dir)
    return input_dir


def outlier_correction(cfg: PipelineConfig) -> Path:
    _require_simba()
    oc = cfg.get("simba.outlier_correction") or {}
    config_path = str(cfg.simba_config_path)
    if oc.get("enabled", True):
        from simba.outlier_tools.outlier_corrector_location import OutlierCorrecterLocation
        from simba.outlier_tools.outlier_corrector_movement import OutlierCorrecterMovement

        # The correctors read keys of the form movement_bodypart1_<animal, lower-cased>.
        values: dict[str, object] = {
            "movement_criterion": float(oc.get("movement_criterion", 0.7)),
            "location_criterion": float(oc.get("location_criterion", 1.5)),
            "mean_or_median": str(oc.get("aggregation", "median")),
        }
        for animal, parts in (oc.get("body_parts") or {}).items():
            key = str(animal).lower()
            for kind in ("Movement", "Location"):
                bps = list(parts.get(kind, []))
                if len(bps) != 2:
                    raise ValueError(f"simba.outlier_correction.body_parts.{animal}.{kind} needs exactly 2 body parts")
                values[f"{kind.lower()}_bodypart1_{key}"] = bps[0]
                values[f"{kind.lower()}_bodypart2_{key}"] = bps[1]
        set_ini(cfg, "Outlier settings", values)
        LOG.info("outlier correction: movement")
        OutlierCorrecterMovement(config_path=config_path).run()
        LOG.info("outlier correction: location")
        OutlierCorrecterLocation(config_path=config_path).run()
    else:
        from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper

        LOG.info("outlier correction skipped (config)")
        OutlierCorrectionSkipper(config_path=config_path).run()
    out = cfg.simba_project_folder / "csv" / "outlier_corrected_movement_location"
    if not list(out.glob("*.csv")):
        raise RuntimeError(f"outlier step produced nothing in {out}")
    return out


def extract_features(cfg: PipelineConfig) -> Path:
    _require_simba()
    from simba.utils.cli.cli_tools import feature_extraction_runner

    LOG.info("extracting features (SimBA 16-bp feature set)")
    feature_extraction_runner(config_path=str(cfg.simba_config_path))
    out = cfg.simba_project_folder / "csv" / "features_extracted"
    files = list(out.glob("*.csv"))
    if not files:
        raise RuntimeError(f"feature extraction produced nothing in {out}")
    LOG.info("%d feature file(s) in %s", len(files), out)
    return out


def build_project(cfg: PipelineConfig, overwrite: bool = False) -> Path:
    """Run every project step in order."""
    create_project(cfg, overwrite=overwrite)
    import_videos(cfg)
    write_video_info(cfg)
    import_pose(cfg)
    outlier_correction(cfg)
    return extract_features(cfg)

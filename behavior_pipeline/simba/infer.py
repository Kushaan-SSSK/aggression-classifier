# Kushaan Sharma
"""Run the trained classifiers on every feature file and aggregate bout statistics.
Run as the `infer` stage of the CLI.
"""
from __future__ import annotations

from pathlib import Path

from ..config import LOG, PipelineConfig
from .project import set_ini


def configure_models(cfg: PipelineConfig, models: dict[str, Path] | None = None) -> dict[str, Path]:
    """Write model paths, thresholds and minimum bout lengths into project_config.ini."""
    generated = cfg.simba_models_dir
    models = models or {clf: generated / f"{clf}.sav" for clf in cfg.classifiers}
    missing = [clf for clf, p in models.items() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"no trained model for {missing} in {generated}; run the train stage")
    thresholds = cfg.get("infer.thresholds") or {}
    min_bouts = cfg.get("infer.min_bout_ms") or {}
    sml: dict[str, object] = {"model_dir": str(generated), "no_targets": len(models)}
    thr: dict[str, object] = {}
    mb: dict[str, object] = {}
    for i, (clf, path) in enumerate(models.items(), start=1):
        sml[f"model_path_{i}"] = str(Path(path).resolve())
        sml[f"target_name_{i}"] = clf
        thr[f"threshold_{i}"] = float(thresholds.get(clf, 0.5))
        mb[f"min_bout_{i}"] = int(min_bouts.get(clf, 0))
    set_ini(cfg, "SML settings", sml)
    set_ini(cfg, "threshold_settings", thr)
    set_ini(cfg, "Minimum_bout_lengths", mb)
    return {clf: Path(p) for clf, p in models.items()}


def run_inference(cfg: PipelineConfig, models: dict[str, Path] | None = None) -> Path:
    from simba.model.inference_batch import InferenceBatch

    configure_models(cfg, models)
    LOG.info("running classifiers on csv/features_extracted")
    InferenceBatch(config_path=str(cfg.simba_config_path)).run()
    out = cfg.simba_project_folder / "csv" / "machine_results"
    files = list(out.glob("*.csv"))
    if not files:
        raise RuntimeError(f"inference produced nothing in {out}")
    LOG.info("%d machine_results file(s)", len(files))
    return out


def aggregate(cfg: PipelineConfig, classifiers: list[str] | None = None) -> None:
    """Per-video bout counts, durations and intervals under project_folder/logs."""
    from simba.data_processors.agg_clf_calculator import AggregateClfCalculator

    classifiers = classifiers or cfg.classifiers
    calc = AggregateClfCalculator(config_path=str(cfg.simba_config_path), classifiers=classifiers,
                                  detailed_bout_data=True, first_occurrence=True, event_count=True,
                                  total_event_duration=True, pct_of_session=True, mean_event_duration=True,
                                  median_event_duration=True, mean_interval_duration=True, median_interval_duration=True,
                                  frame_count=True, video_length=True)
    calc.run()
    calc.save()
    LOG.info("aggregate classifier statistics saved under project_folder/logs")


def infer_all(cfg: PipelineConfig, models: dict[str, Path] | None = None) -> Path:
    out = run_inference(cfg, models)
    try:
        aggregate(cfg, list(models.keys()) if models else None)
    except Exception as exc:
        LOG.warning("aggregate statistics failed: %s", exc)
    return out

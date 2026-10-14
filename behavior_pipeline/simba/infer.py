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
    if models is None:
        # default: every configured behaviour that has a trained model
        models = {clf: generated / f"{clf}.sav" for clf in cfg.classifiers if (generated / f"{clf}.sav").exists()}
        untrained = [clf for clf in cfg.classifiers if clf not in models]
        if untrained:
            LOG.info("no model for %s, skipped", untrained)
    missing = [clf for clf, p in models.items() if not Path(p).exists()]
    if missing or not models:
        raise FileNotFoundError(f"no trained model for {missing or cfg.classifiers} in {generated}; run the train stage")
    thresholds = cfg.get("infer.thresholds") or {}
    min_bouts = cfg.get("infer.min_bout_ms") or {}
    # SimBA saves new models under <model_dir>/generated_models, so model_dir stays the parent
    sml: dict[str, object] = {"model_dir": str(generated.parent), "no_targets": len(models)}
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

    from .project import _ensure_classifiers, _read_ini

    ini = _read_ini(cfg.simba_config_path)
    registered = [ini.get("SML settings", k) for k in ini["SML settings"] if k.startswith("target_name_")]
    configure_models(cfg, models)
    LOG.info("running classifiers on csv/features_extracted")
    try:
        InferenceBatch(config_path=str(cfg.simba_config_path)).run()
    finally:
        # inference narrows the ini to the models run; put the other classifiers back for training
        _ensure_classifiers(cfg, extra=registered)
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


def smooth_probs(prob, fps: float, window_ms: float):
    """Centred rolling mean of a probability trace; window_ms <= 0 returns it unchanged."""
    import numpy as np
    import pandas as pd

    p = np.asarray(prob, dtype=float)
    win = int(round(window_ms / 1000.0 * fps))
    if win <= 1 or p.size == 0:
        return p
    return pd.Series(p).rolling(win, center=True, min_periods=1).mean().to_numpy()


def join_gaps(mask, fps: float, gap_ms: float):
    """Fill pauses shorter than gap_ms between two positive runs; gap_ms <= 0 returns the mask unchanged."""
    import numpy as np

    m = np.asarray(mask, dtype=bool).copy()
    gap = int(round(gap_ms / 1000.0 * fps))
    if gap <= 0 or m.size == 0:
        return m
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    for prev_end, next_start in zip(ends[:-1], starts[1:]):
        if next_start - prev_end < gap:
            m[prev_end:next_start] = True
    return m


def viterbi_binary(prob, threshold: float = 0.5, penalty: float = 0.0):
    """Two-state (absent / present) decoding of a probability trace.

    Each frame scores log(p / threshold) for present and log((1 - p) / (1 - threshold)) for absent, and every
    change of state costs `penalty` (in log units). With penalty 0 this is a plain threshold; larger penalties
    remove flicker without the blurring of a moving average.
    """
    import numpy as np

    p = np.clip(np.asarray(prob, dtype=float), 1e-4, 1 - 1e-4)
    n = p.size
    if n == 0 or penalty <= 0:
        return p >= threshold
    thr = min(max(float(threshold), 1e-4), 1 - 1e-4)
    e = np.stack([np.log((1 - p) / (1 - thr)), np.log(p / thr)], axis=1)
    back = np.zeros((n, 2), dtype=np.int8)
    score = e[0].copy()
    for t in range(1, n):
        stay, switch = score, score[::-1] - penalty
        back[t] = np.where(stay >= switch, [0, 1], [1, 0])
        score = np.maximum(stay, switch) + e[t]
    states = np.zeros(n, dtype=np.int8)
    states[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states.astype(bool)


def keep_peaks(mask, prob, threshold: float, peak: float):
    """Add back short events the Viterbi decode removed: every run of prob >= threshold whose maximum reaches
    `peak` is kept whole. A switch penalty that stops flicker also deletes 2-6 frame bites (an isolated bout must
    win back two switch costs), so confident peaks are restored after decoding; peak <= 0 returns mask unchanged."""
    import numpy as np

    m = np.asarray(mask, dtype=bool).copy()
    p = np.asarray(prob, dtype=float)
    if peak <= 0 or p.size == 0:
        return m
    d = np.diff(np.concatenate([[0], (p >= threshold).astype(int), [0]]))
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if p[a:b].max() >= peak:
            m[a:b] = True
    return m


def postprocess_machine_results(cfg: PipelineConfig, classifiers: list[str] | None = None) -> list[str]:
    """Smooth Probability_<clf>, re-threshold (Viterbi when infer.switch_penalty is set), join short gaps and
    re-apply the minimum bout in machine_results.

    Only classifiers with a non-zero infer.smooth_ms or infer.join_gap_ms are touched. SimBA applied its
    own threshold and minimum bout before smoothing, so both are redone here on the smoothed trace.
    """
    import pandas as pd

    from .annotations import _video_fps
    from .proximity import min_bout_filter

    classifiers = classifiers or cfg.classifiers
    smooth = cfg.get("infer.smooth_ms") or {}
    join = cfg.get("infer.join_gap_ms") or {}
    thresholds = cfg.get("infer.thresholds") or {}
    min_bouts = cfg.get("infer.min_bout_ms") or {}
    penalty = cfg.get("infer.switch_penalty") or {}
    peaks = cfg.get("infer.peak_threshold") or {}
    active = [c for c in classifiers if any(float(d.get(c, 0) or 0) > 0 for d in (smooth, join, penalty, peaks))]
    if not active:
        return []
    fps_by_video = _video_fps(cfg)
    results = cfg.simba_project_folder / "csv" / "machine_results"
    for f in sorted(results.glob("*.csv")):
        df = pd.read_csv(f, index_col=0)
        fps = fps_by_video.get(f.stem, cfg.fps)
        for clf in active:
            col = f"Probability_{clf}"
            if col not in df.columns:
                continue
            p = smooth_probs(df[col].to_numpy(), fps, float(smooth.get(clf, 0) or 0))
            m = viterbi_binary(p, float(thresholds.get(clf, 0.5)), float(penalty.get(clf, 0) or 0))
            m = keep_peaks(m, p, float(thresholds.get(clf, 0.5)), float(peaks.get(clf, 0) or 0))
            m = join_gaps(m, fps, float(join.get(clf, 0) or 0))
            m = min_bout_filter(m, int(round(float(min_bouts.get(clf, 0) or 0) / 1000.0 * fps)))
            df[col] = p
            df[clf] = m.astype(int)
        df.to_csv(f)
    LOG.info("post-processed %s in machine_results (smooth %s ms, switch penalty %s, peaks %s, join gaps %s ms)", active,
             {c: smooth.get(c, 0) for c in active}, {c: penalty.get(c, 0) for c in active}, {c: peaks.get(c, 0) for c in active},
             {c: join.get(c, 0) for c in active})
    return active


def trained_classifiers(cfg: PipelineConfig) -> list[str]:
    return [clf for clf in cfg.classifiers if (cfg.simba_models_dir / f"{clf}.sav").exists()]


def infer_all(cfg: PipelineConfig, models: dict[str, Path] | None = None) -> Path:
    out = run_inference(cfg, models)
    clfs = list(models.keys()) if models else trained_classifiers(cfg)
    postprocess_machine_results(cfg, clfs)
    try:
        aggregate(cfg, clfs)
    except Exception as exc:
        LOG.warning("aggregate statistics failed: %s", exc)
    return out


def bouts(mask, fps: float) -> list[tuple[float, float]]:
    """(start_s, end_s) runs of ones in a 0/1 sequence."""
    import numpy as np

    m = np.asarray(mask, dtype=int)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s / fps, e / fps) for s, e in zip(starts, ends)]


def compare_to_annotations(cfg: PipelineConfig, classifier: str = "Attack") -> "pd.DataFrame":
    """Per video: predicted frames and bouts against the annotated ones; writes logs/<clf>_vs_annotations.csv."""
    import pandas as pd

    fps = cfg.fps
    results = cfg.simba_project_folder / "csv" / "machine_results"
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    rows = []
    bout_rows = []
    for f in sorted(results.glob("*.csv")):
        pred = pd.read_csv(f, usecols=lambda c: c in (classifier, f"Probability_{classifier}"))
        p = pred[classifier].astype(int).to_numpy()
        t_file = targets / f.name
        annotated = t_file.exists()
        t = pd.read_csv(t_file, usecols=[classifier])[classifier].astype(int).to_numpy() if annotated else None
        pb = bouts(p, fps)
        rec = {"video": f.stem, "frames": len(p), "annotated": annotated, "pred_frames": int(p.sum()), "pred_bouts": len(pb),
               "pred_s": round(p.sum() / fps, 1), "max_prob": round(float(pred[f"Probability_{classifier}"].max()), 3)}
        if annotated:
            n = min(len(p), len(t))
            p, t = p[:n], t[:n]
            tp = int((p & t).sum())
            tb = bouts(t, fps)
            hit = sum(any(bs < te and be > ts for bs, be in pb) for ts, te in tb)
            rec.update({"true_frames": int(t.sum()), "true_bouts": len(tb), "true_bouts_hit": hit,
                        "precision": round(tp / p.sum(), 3) if p.sum() else None, "recall": round(tp / t.sum(), 3) if t.sum() else None})
        rows.append(rec)
        for s, e in pb:
            bout_rows.append({"video": f.stem, "start_s": round(s, 2), "end_s": round(e, 2), "duration_s": round(e - s, 2),
                              "overlaps_annotation": bool(annotated and t[int(s * fps):int(e * fps)].any())})
    logs = cfg.simba_project_folder / "logs"
    table = pd.DataFrame(rows)
    table.to_csv(logs / f"{classifier}_vs_annotations.csv", index=False)
    pd.DataFrame(bout_rows).to_csv(logs / f"{classifier}_predicted_bouts.csv", index=False)
    LOG.info("%s predictions vs annotations:\n%s", classifier, table.to_string(index=False))
    return table

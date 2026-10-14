# Kushaan Sharma
"""Cross-validation of a behaviour and operating-point sweeps on the out-of-fold probabilities.
Run as the `evaluate` stage of the CLI; the numbers land under project_folder/logs.

Two splits:
  shuffled  SimBA's reporting standard (Goodwin et al. 2024): every labelled frame pooled and split k ways at
            random. Neighbouring frames of one bout land on both sides, so the scores run high.
  held-out  each resident's sessions held out in turn; what to expect on a new cohort.
The model card (logs/<clf>_model_card.json) reports both, SimBA's first.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig
from .annotations import _video_fps, read_mask
from .infer import bouts, join_gaps, keep_peaks, smooth_probs, viterbi_binary
from .proximity import min_bout_filter
from .train import _target_columns, algorithm, ensemble_settings, make_estimator, undersample

_ELS = re.compile(r"(ELS\d+)", re.IGNORECASE)


def resident_groups(cfg: PipelineConfig, names: list[str], group_by: str | None = None) -> dict[str, str]:
    """video_name -> fold. Sessions of one resident (same ELS id) share a fold unless group_by is 'video'."""
    group_by = group_by or str(cfg.get("evaluate.group_by", "resident"))
    override = {str(k): str(v) for k, v in (cfg.get("evaluate.resident_groups") or {}).items()}
    out = {}
    for n in names:
        if n in override:
            out[n] = override[n]
            continue
        m = _ELS.search(n)
        out[n] = m.group(1).upper() if (m and group_by == "resident") else n
    return out


def pose_columns(cfg: PipelineConfig) -> list[str]:
    """The body-part coordinate columns SimBA drops before training."""
    try:
        from simba.mixins.config_reader import ConfigReader

        reader = ConfigReader(config_path=str(cfg.simba_config_path), read_video_info=False, create_logger=False)
        return list(reader.bp_col_names)
    except Exception as exc:  # no SimBA project yet: rebuild the names from the keypoint map
        LOG.debug("ConfigReader unavailable (%s); deriving pose columns from the config", exc)
        parts = list((cfg.get("pose.keypoint_map") or {}).get("superanimal", {}).keys())
        animals = int(cfg.get("simba.animal_cnt", 2))
        return [f"{bp}_{a}_{s}" for a in range(1, animals + 1) for bp in parts for s in ("x", "y", "p")]


def load_targets(cfg: PipelineConfig, classifier: str, names: list[str] | None = None) -> dict[str, dict]:
    """Per video: X (features, float32), y (0/1) and known (label mask) for one classifier."""
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    drop = set(pose_columns(cfg)) | set(_target_columns(cfg)) | set(cfg.classifiers)
    out = {}
    for f in sorted(targets.glob("*.csv")):
        if names and f.stem not in names:
            continue
        df = pd.read_csv(f, index_col=0)
        if classifier not in df.columns:
            continue
        feats = [c for c in df.columns if c not in drop]
        known = read_mask(cfg, f.stem, classifier, n=len(df))
        known = np.ones(len(df), dtype=bool) if known is None else known.to_numpy().astype(bool)
        if len(known) < len(df):
            known = np.concatenate([known, np.ones(len(df) - len(known), dtype=bool)])
        out[f.stem] = {"X": df[feats].fillna(0).to_numpy(np.float32), "y": df[classifier].to_numpy().astype(int), "known": known,
                       "features": feats}
    return out


def _fit(cfg: PipelineConfig, classifier: str, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, features: list[str] | None = None):
    idx = undersample(cfg, classifier, y, rng)
    if algorithm(cfg, classifier) == "ET":
        return make_estimator(cfg, classifier, features or []).fit(X[idx], y[idx])
    from sklearn.ensemble import RandomForestClassifier

    s = ensemble_settings(cfg, classifier)
    cw = None if str(s["class_weights"]).lower() == "none" else str(s["class_weights"])
    rf = RandomForestClassifier(n_estimators=int(s["rf_n_estimators"]), max_features=s["rf_max_features"], criterion=s["rf_criterion"],
                                min_samples_leaf=int(s["rf_min_sample_leaf"]), class_weight=cw, n_jobs=int(cfg.get("train.n_jobs", -1)),
                                random_state=int(cfg.get("evaluate.seed", 0)))
    return rf.fit(X[idx], y[idx])


def _bout_list(y: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, y.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _video_metrics(pred: np.ndarray, y: np.ndarray, known: np.ndarray, fps: float) -> dict:
    p, t = pred.astype(int) & known, y.astype(int) & known
    tp, fp, fn = int((p & t).sum()), int((p & (1 - t)).sum()), int(((1 - p) & t).sum())
    tb, pb = _bout_list(t), _bout_list(p)
    hit = sum(1 for a, b in tb if p[a:b].any())
    false = sum(1 for a, b in pb if not t[a:b].any())
    return {"frames": int(len(y)), "known_frames": int(known.sum()), "true_frames": int(t.sum()), "pred_frames": int(p.sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
            "true_bouts": len(tb), "true_bouts_hit": hit, "pred_bouts": len(pb), "false_bouts": false, "fp_seconds": round(fp / fps, 1)}


def dilate(mask: np.ndarray, frames: int) -> np.ndarray:
    """Grow every positive run by `frames` on both sides."""
    m = np.asarray(mask, dtype=bool)
    if frames <= 0 or not m.any():
        return m.copy()
    return np.convolve(m.astype(int), np.ones(2 * frames + 1, dtype=int), mode="same") > 0


def tolerant_counts(pred: np.ndarray, y: np.ndarray, known: np.ndarray, frames: int) -> dict:
    """Frame counts that forgive timing errors up to `frames`: a predicted frame is correct when it lies within
    `frames` of a labelled frame, a labelled frame is found when a prediction lies within `frames` of it.
    Nothing is excluded, so a bout the model misses entirely still counts in full."""
    p, t = np.asarray(pred, bool) & known, np.asarray(y, bool) & known
    return {"pred": int(p.sum()), "pred_ok": int((p & dilate(t, frames)).sum()),
            "true": int(t.sum()), "true_found": int((t & dilate(p, frames)).sum())}


def segment_counts(pred: np.ndarray, y: np.ndarray, known: np.ndarray, iou: float) -> tuple[int, int, int]:
    """Segmental F1 counts (Lea et al. 2017): a predicted bout is a hit when it overlaps an unmatched labelled bout
    with intersection-over-union >= iou. Returns (hits, predicted bouts, labelled bouts)."""
    tb = _bout_list(np.asarray(y, bool) & known)
    pb = _bout_list(np.asarray(pred, bool) & known)
    used: set[int] = set()
    hits = 0
    for a, b in pb:
        best, best_j = 0.0, None
        for j, (c, d) in enumerate(tb):
            if j in used or d <= a or c >= b:
                continue
            ov = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if ov > best:
                best, best_j = ov, j
        if best_j is not None and best >= iou:
            used.add(best_j)
            hits += 1
    return hits, len(pb), len(tb)


def episode_counts(pred: np.ndarray, y: np.ndarray, known: np.ndarray, gap_frames: int) -> tuple[int, int, int]:
    """Event counts on attack episodes: labelled and predicted bouts closer than gap_frames are merged first
    (scorers split one fight at pauses differently), then each predicted episode may claim at most one
    overlapping labelled episode (the largest overlap). Returns (matched, predicted episodes, labelled episodes)."""
    def merged(m):
        m = np.asarray(m, bool).copy()
        bl = _bout_list(m)
        for (_, b1), (a2, _) in zip(bl, bl[1:]):
            if a2 - b1 <= gap_frames:
                m[b1:a2] = True
        return _bout_list(m)

    tb, pb = merged(np.asarray(y, bool) & known), merged(np.asarray(pred, bool) & known)
    used: set[int] = set()
    for a, b in pb:
        ov = [(min(b, d) - max(a, c), j) for j, (c, d) in enumerate(tb) if j not in used and min(b, d) > max(a, c)]
        if ov:
            used.add(max(ov)[1])
    return len(used), len(pb), len(tb)


SPLITS = {"held-out": "lovo", "shuffled": "shuffled"}


def probs_dir(cfg: PipelineConfig, classifier: str, split: str = "held-out") -> Path:
    return cfg.simba_project_folder / "logs" / f"{classifier}_{SPLITS[split]}_probs"


def _fold_done(cfg: PipelineConfig, classifier: str, videos: list[str]) -> bool:
    """True when every video of a fold has held-out probabilities newer than its targets file."""
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    for v in videos:
        p = probs_dir(cfg, classifier) / f"{v}.csv"
        t = targets / f"{v}.csv"
        if not p.exists() or not t.exists() or p.stat().st_mtime < t.stat().st_mtime:
            return False
    return True


def lovo_table(cfg: PipelineConfig, classifier: str, threshold: float = 0.5, group_by: str | None = None) -> pd.DataFrame:
    """Per-video table from the saved held-out probabilities; written to logs/<clf>_leave_one_video_out.csv."""
    probs = load_probs(cfg, classifier)
    groups = resident_groups(cfg, list(probs), group_by)
    fps_by_video = _video_fps(cfg)
    rows = []
    for v, df in probs.items():
        fps = fps_by_video.get(v, cfg.fps)
        y, known = df["y"].to_numpy().astype(int), df["known"].to_numpy().astype(bool)
        rows.append({"video": v, "group": groups[v], **_video_metrics(df["prob"].to_numpy() >= threshold, y, known, fps)})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.simba_project_folder / "logs" / f"{classifier}_leave_one_video_out.csv", index=False)
    return table


def run_lovo(cfg: PipelineConfig, classifier: str, group_by: str | None = None, names: list[str] | None = None,
             threshold: float = 0.5, resume: bool = False) -> pd.DataFrame:
    """Hold each fold out in turn, train on the rest, save the held-out per-frame probabilities.

    With resume, folds whose probabilities are already newer than their targets are skipped.
    """
    data = load_targets(cfg, classifier, names)
    if not data:
        raise FileNotFoundError(f"no targets with a {classifier} column; run annotate first")
    groups = resident_groups(cfg, list(data), group_by)
    rng = np.random.default_rng(int(cfg.get("evaluate.seed", 0)))
    out_dir = probs_dir(cfg, classifier)
    out_dir.mkdir(parents=True, exist_ok=True)
    for g in sorted(set(groups.values())):
        test = [v for v in data if groups[v] == g]
        train = [v for v in data if groups[v] != g]
        if resume and _fold_done(cfg, classifier, test):
            LOG.info("fold %s already done, skipped", g)
            continue
        xt = np.vstack([data[v]["X"][data[v]["known"]] for v in train])
        yt = np.concatenate([data[v]["y"][data[v]["known"]] for v in train])
        if yt.sum() == 0 or yt.sum() == len(yt):
            LOG.warning("fold %s: training data has a single class, skipped", g)
            continue
        t0 = time.time()
        rf = _fit(cfg, classifier, xt, yt, rng, next(iter(data.values()))["features"])
        fit_s = round(time.time() - t0)
        for v in test:
            d = data[v]
            prob = rf.predict_proba(d["X"])[:, 1]
            pd.DataFrame({"frame_idx": np.arange(len(prob)), "prob": prob, "y": d["y"], "known": d["known"].astype(int)}).to_csv(out_dir / f"{v}.csv", index=False)
            LOG.info("fold %s (%d s): %s", g, fit_s, {"video": v, **_video_metrics(prob >= threshold, d["y"], d["known"], cfg.fps)})
    return lovo_table(cfg, classifier, threshold, group_by)


def run_shuffled(cfg: PipelineConfig, classifier: str, names: list[str] | None = None, folds: int | None = None) -> Path:
    """SimBA-style k-fold cross-validation on shuffled frames: every labelled frame of every video is pooled and
    assigned to a fold at random, so each frame gets an out-of-fold probability (saved per video like the held-out
    split, for the same metrics)."""
    data = load_targets(cfg, classifier, names)
    if not data:
        raise FileNotFoundError(f"no targets with a {classifier} column; run annotate first")
    folds = int(folds or cfg.get("evaluate.shuffle_folds", 5))
    rng = np.random.default_rng(int(cfg.get("evaluate.seed", 0)))
    vids = list(data)
    X = np.vstack([data[v]["X"] for v in vids])
    y = np.concatenate([data[v]["y"] for v in vids])
    known = np.concatenate([data[v]["known"] for v in vids])
    owner = np.repeat(np.arange(len(vids)), [len(data[v]["y"]) for v in vids])
    fold = rng.integers(0, folds, len(y))
    prob = np.zeros(len(y), dtype=np.float32)
    features = data[vids[0]]["features"]
    for f in range(folds):
        t0 = time.time()
        train = (fold != f) & known
        model = _fit(cfg, classifier, X[train], y[train], rng, features)
        test = fold == f
        prob[test] = model.predict_proba(X[test])[:, 1]
        LOG.info("%s shuffled-frame fold %d/%d (%d s)", classifier, f + 1, folds, round(time.time() - t0))
    out_dir = probs_dir(cfg, classifier, "shuffled")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(vids):
        sel = owner == i
        pd.DataFrame({"frame_idx": np.arange(sel.sum()), "prob": prob[sel], "y": y[sel], "known": known[sel].astype(int)}).to_csv(
            out_dir / f"{v}.csv", index=False)
    return out_dir


def load_probs(cfg: PipelineConfig, classifier: str, split: str = "held-out") -> dict[str, pd.DataFrame]:
    d = probs_dir(cfg, classifier, split)
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    files = sorted(d.glob("*.csv"))
    current = {f.stem for f in targets.glob("*.csv")}
    if current:
        stale = [f.stem for f in files if f.stem not in current]
        if stale:
            LOG.info("%s: ignoring held-out probabilities of videos without targets: %s", classifier, stale)
        files = [f for f in files if f.stem in current]
    if not files:
        raise FileNotFoundError(f"no out-of-fold probabilities in {d}; run evaluate without --sweep-only first")
    return {f.stem: pd.read_csv(f) for f in files}


def score(cfg: PipelineConfig, probs: dict[str, pd.DataFrame], preds: dict[str, np.ndarray], fps_by_video: dict[str, float],
          tols_ms: list[float] = (), seg_ious: list[float] = (), episode_gap_ms: float = 1000.0) -> dict:
    """Pooled frame, tolerant, segmental, episode and bout scores of decoded predictions against the labels."""
    tot = {"hit": 0, "bouts": 0, "false": 0, "tp": 0, "fp": 0, "fn": 0, "neg_s": 0.0, "neg_bouts": 0, "neg_min": 0.0}
    tolc = {t: {"pred": 0, "pred_ok": 0, "true": 0, "true_found": 0} for t in tols_ms}
    segc = {k: [0, 0, 0] for k in seg_ious}
    epc = [0, 0, 0]
    for v, df in probs.items():
        fps = fps_by_video.get(v, cfg.fps)
        y, known = df["y"].to_numpy().astype(int), df["known"].to_numpy().astype(bool)
        pred = preds[v]
        m = _video_metrics(pred, y, known, fps)
        tot["hit"] += m["true_bouts_hit"]; tot["bouts"] += m["true_bouts"]; tot["false"] += m["false_bouts"]
        tot["tp"] += m["tp"]; tot["fp"] += m["fp"]; tot["fn"] += m["fn"]
        for t in tols_ms:
            for k, n in tolerant_counts(pred, y, known, int(round(t / 1000.0 * fps))).items():
                tolc[t][k] += n
        for k in seg_ious:
            segc[k] = [a + b for a, b in zip(segc[k], segment_counts(pred, y, known, k))]
        epc = [a + b for a, b in zip(epc, episode_counts(pred, y, known, int(round(episode_gap_ms / 1000.0 * fps))))]
        if m["true_frames"] == 0 and known.sum():
            tot["neg_s"] += m["pred_frames"] / fps; tot["neg_bouts"] += m["pred_bouts"]; tot["neg_min"] += known.sum() / fps / 60
    prec = tot["tp"] / max(tot["tp"] + tot["fp"], 1)
    rec = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    out = {"bouts_hit": f"{tot['hit']}/{tot['bouts']}", "bout_recall": round(tot["hit"] / max(tot["bouts"], 1), 2),
           "false_bouts_all": tot["false"],
           "false_bouts_per_min_on_negatives": round(tot["neg_bouts"] / tot["neg_min"], 2) if tot["neg_min"] else None,
           "false_s_per_min_on_negatives": round(tot["neg_s"] / tot["neg_min"], 2) if tot["neg_min"] else None,
           "frame_precision": round(prec, 2), "frame_recall": round(rec, 2), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 2)}
    for t, c in tolc.items():
        pt = c["pred_ok"] / max(c["pred"], 1)
        rt = c["true_found"] / max(c["true"], 1)
        tag = f"{t / 1000:g}s"
        out.update({f"frame_precision_tol{tag}": round(pt, 2), f"frame_recall_tol{tag}": round(rt, 2),
                    f"f1_tol{tag}": round(2 * pt * rt / max(pt + rt, 1e-9), 2)})
    for k, (h, npred, ntrue) in segc.items():
        out[f"segment_f1@{int(k * 100)}"] = round(2 * h / max(npred + ntrue, 1), 2)
    out.update({"episodes_found": f"{epc[0]}/{epc[2]}", "episode_precision": round(epc[0] / max(epc[1], 1), 2),
                "episode_recall": round(epc[0] / max(epc[2], 1), 2), "episode_f1": round(2 * epc[0] / max(epc[1] + epc[2], 1), 2)})
    return out


def sweep(cfg: PipelineConfig, classifier: str, thresholds=None, min_bout_ms=None, smooth_ms=None, join_gap_ms=None,
          switch_penalty=None, peak_threshold=None, split: str = "held-out") -> pd.DataFrame:
    """Score every smoothing x threshold x switch-penalty x peak x gap-join x minimum-bout combination on the out-of-fold probabilities."""
    ev = cfg.get("evaluate") or {}
    thresholds = list(thresholds or ev.get("thresholds") or [0.5])
    min_bout_ms = list(min_bout_ms or ev.get("min_bout_ms") or [0])
    smooth_ms = list(smooth_ms or ev.get("smooth_ms") or [0])
    join_gap_ms = list(join_gap_ms or ev.get("join_gap_ms") or [0])
    penalties = list(switch_penalty or ev.get("switch_penalty") or [0])
    peak_list = list(peak_threshold or ev.get("peak_threshold") or [0])
    episode_gap_ms = float(ev.get("episode_gap_ms", 1000) or 0)
    probs = load_probs(cfg, classifier, split)
    fps_by_video = _video_fps(cfg)
    tol = ev.get("edge_tolerance_ms", 0) or 0
    tols_ms = [float(t) for t in (tol if isinstance(tol, (list, tuple)) else [tol]) if float(t) > 0]
    seg_ious = [float(x) for x in (ev.get("segment_iou") or [])]
    rows = []
    for sm in smooth_ms:
        smoothed = {v: smooth_probs(df["prob"].to_numpy(), fps_by_video.get(v, cfg.fps), float(sm)) for v, df in probs.items()}
        for thr, pen, pk in [(t, q, k) for t in thresholds for q in penalties for k in peak_list]:
            if pk and pk <= thr:
                continue
            decoded = {v: keep_peaks(viterbi_binary(smoothed[v], float(thr), float(pen)), smoothed[v], float(thr), float(pk))
                       for v in probs}
            for gap in join_gap_ms:
                for mb in min_bout_ms:
                    preds = {}
                    for v in probs:
                        fps = fps_by_video.get(v, cfg.fps)
                        preds[v] = min_bout_filter(join_gaps(decoded[v], fps, float(gap)), int(round(float(mb) / 1000.0 * fps)))
                    rows.append({"smooth_ms": sm, "threshold": thr, "switch_penalty": pen, "peak_threshold": pk, "join_gap_ms": gap,
                                 "min_bout_ms": mb, **score(cfg, probs, preds, fps_by_video, tols_ms, seg_ious, episode_gap_ms)})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.simba_project_folder / "logs" / f"{classifier}_operating_points_{SPLITS[split]}.csv", index=False)
    return table


def deployed_decoder(cfg: PipelineConfig, classifier: str) -> dict[str, float]:
    """The infer stage's post-processing for one classifier, read from infer.*."""
    def get(key: str, default: float = 0.0) -> float:
        return float((cfg.get(f"infer.{key}") or {}).get(classifier, default) or default)

    return {"threshold": get("thresholds", 0.5), "smooth_ms": get("smooth_ms"), "switch_penalty": get("switch_penalty"),
            "peak_threshold": get("peak_threshold"), "join_gap_ms": get("join_gap_ms"), "min_bout_ms": get("min_bout_ms")}


def decode(prob: np.ndarray, fps: float, d: dict[str, float]) -> np.ndarray:
    """Apply one decoder (deployed_decoder's keys) to a probability trace, in the infer stage's order."""
    p = smooth_probs(prob, fps, d["smooth_ms"])
    m = keep_peaks(viterbi_binary(p, d["threshold"], d["switch_penalty"]), p, d["threshold"], d["peak_threshold"])
    return min_bout_filter(join_gaps(m, fps, d["join_gap_ms"]), int(round(d["min_bout_ms"] / 1000.0 * fps)))


def model_card(cfg: PipelineConfig, classifier: str) -> dict:
    """Headline numbers for every split with saved probabilities; written to logs/<clf>_model_card.json.

    simba_standard: frame precision / recall / F1 at threshold 0.5 with no post-processing (SimBA's classification
    report). deployed: the infer stage's decoder, with the boundary-tolerant, episode and bout scores."""
    fps_by_video = _video_fps(cfg)
    dec = deployed_decoder(cfg, classifier)
    gap = float(cfg.get("evaluate.episode_gap_ms", 1000) or 0)
    card = {"classifier": classifier, "created": time.strftime("%Y-%m-%d %H:%M"), "decoder": dec, "splits": {}}
    model = cfg.simba_models_dir / f"{classifier}.sav"
    if model.exists():
        card["model_trained"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(model.stat().st_mtime))
    for split in ("shuffled", "held-out"):
        try:
            probs = load_probs(cfg, classifier, split)
        except FileNotFoundError:
            continue
        plain = {v: df["prob"].to_numpy() >= 0.5 for v, df in probs.items()}
        deployed = {v: decode(df["prob"].to_numpy(), fps_by_video.get(v, cfg.fps), dec) for v, df in probs.items()}
        s = score(cfg, probs, plain, fps_by_video)
        newest = max((probs_dir(cfg, classifier, split) / f"{v}.csv").stat().st_mtime for v in probs)
        card["splits"][split] = {
            "method": (f"{int(cfg.get('evaluate.shuffle_folds', 5))}-fold cross-validation on shuffled frames (SimBA standard)"
                       if split == "shuffled" else f"leave-one-{cfg.get('evaluate.group_by', 'resident')}-out cross-validation"),
            "videos": len(probs), "groups": len(set(resident_groups(cfg, list(probs)).values())),
            "labelled_frames": sum(int(df["known"].sum()) for df in probs.values()),
            "positive_frames": sum(int((df["y"].astype(bool) & df["known"].astype(bool)).sum()) for df in probs.values()),
            "probs_saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)),
            "simba_standard": {k: s[k] for k in ("frame_precision", "frame_recall", "f1")},
            "deployed": score(cfg, probs, deployed, fps_by_video, [1000.0], [], gap)}
    (cfg.simba_project_folder / "logs" / f"{classifier}_model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def card_summary(card: dict) -> str:
    """Plain-text headline of a model card, SimBA's split first."""
    lines = [f"{card['classifier']}:"]
    for c in card["splits"].values():
        s, d = c["simba_standard"], c["deployed"]
        lines.append(f"  {c['method']} ({c['videos']} videos): F1 {s['f1']:.2f} (precision {s['frame_precision']:.2f}, "
                     f"recall {s['frame_recall']:.2f}) at threshold 0.5; with the deployed decoder F1 {d['f1']:.2f}, "
                     f"within 1 s {d['f1_tol1s']:.2f}, episode F1 {d['episode_f1']:.2f} ({d['episodes_found']})")
    return "\n".join(lines)


def evaluate(cfg: PipelineConfig, classifiers: list[str], sweep_only: bool = False, group_by: str | None = None,
             names: list[str] | None = None, resume: bool = False, splits: list[str] | None = None, **grid) -> dict[str, pd.DataFrame]:
    """Run the requested splits (default evaluate.splits), sweep operating points on the held-out one and write the model card.
    The shuffled split is only reported: operating points are always chosen on held-out residents."""
    splits = list(splits or cfg.get("evaluate.splits") or ["shuffled", "held-out"])
    out = {}
    for clf in classifiers:
        if "shuffled" in splits and not sweep_only:
            run_shuffled(cfg, clf, names)
        if "held-out" in splits:
            if not sweep_only:
                lovo = run_lovo(cfg, clf, group_by, names, resume=resume)
                LOG.info("%s leave-one-out:\n%s", clf, lovo.to_string(index=False))
            table = sweep(cfg, clf, **grid)
            pd.set_option("display.width", 250)
            LOG.info("%s operating points:\n%s", clf, table.to_string(index=False))
            out[clf] = table
        LOG.info("%s", card_summary(model_card(cfg, clf)))
    return out

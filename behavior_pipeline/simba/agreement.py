# Kushaan Sharma
"""Agreement between two annotators, and between the model and each of them, on the videos both scored.
Run as the `agreement` stage of the CLI: the second scorer's workbook is converted to its own CSV (the training
labels are not touched) and compared with the main annotation CSV. Results in logs/agreement_<clf>.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig
from ..video.manifest import read_annotations
from .annotations import _frame_range


def label_vectors(cfg: PipelineConfig, csv: Path, classifier: str, lengths: dict[str, int], fps: dict[str, float]) -> dict[str, np.ndarray]:
    """0/1 frame vector per video for one behaviour; a video with a NONE row is fully scored and empty."""
    out: dict[str, np.ndarray] = {}
    none = str(cfg.get("annotations.none_token", "NONE")).upper()
    for r in read_annotations(csv, cfg):
        n = lengths.get(r.video_name)
        if n is None:
            continue
        v = out.setdefault(r.video_name, np.zeros(n, dtype=bool))
        if r.behavior == classifier and r.behavior != none and r.start_s is not None:
            a, b = _frame_range(r.start_s, r.end_s, fps.get(r.video_name, cfg.fps), n)
            v[a:b] = True
    return out


def compare(a: np.ndarray, b: np.ndarray, fps: float) -> dict:
    """b scored against a as the reference: frame F1 (strict and within 1 s), bouts of a found by b and vice versa."""
    from .evaluate import _bout_list, segment_counts, tolerant_counts

    known = np.ones(len(a), dtype=bool)
    tp, fp, fn = int((a & b).sum()), int((~a & b).sum()), int((a & ~b).sum())
    tol = tolerant_counts(b, a, known, int(round(fps)))
    ba, bb = _bout_list(a), _bout_list(b)
    return {"frames_a": int(a.sum()), "frames_b": int(b.sum()), "tp": tp, "fp": fp, "fn": fn,
            "bouts_a": len(ba), "bouts_a_found_by_b": sum(1 for s, e in ba if b[s:e].any()),
            "bouts_b": len(bb), "bouts_b_found_by_a": sum(1 for s, e in bb if a[s:e].any()),
            "tol_pred": tol["pred"], "tol_pred_ok": tol["pred_ok"], "tol_true": tol["true"], "tol_true_found": tol["true_found"],
            "seg": segment_counts(b, a, known, 0.5)}


def _summary(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0] if k != "seg"}
    seg = np.sum([r["seg"] for r in rows], axis=0)
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    pt, rt = s["tol_pred_ok"] / max(s["tol_pred"], 1), s["tol_true_found"] / max(s["tol_true"], 1)
    return {"frame_f1": round(f1(s["tp"], s["fp"], s["fn"]), 3), "frame_f1_within_1s": round(2 * pt * rt / max(pt + rt, 1e-9), 3),
            "bouts_of_ref_found": f"{s['bouts_a_found_by_b']}/{s['bouts_a']}", "bouts_found_in_ref": f"{s['bouts_b_found_by_a']}/{s['bouts_b']}",
            "segment_f1@50": round(2 * seg[0] / max(seg[1] + seg[2], 1), 3)}


def run_agreement(cfg: PipelineConfig, xlsx: str | Path, annotator: str, classifier: str = "Attack",
                  threshold: float | None = None, penalty: float | None = None) -> pd.DataFrame:
    from .annotations import _video_fps, source_fps_by_name, workbook_to_csv
    from .evaluate import load_probs
    from .infer import viterbi_binary
    from ..video.manifest import behavior_aliases

    second = cfg.path("paths.annotations").parent / f"second_rater_{annotator or 'B'}.csv"
    workbook_to_csv(xlsx, second, annotator=annotator, merge=True, aliases=behavior_aliases(cfg),
                    sheet_renames=cfg.get("annotations.sheet_renames"), source_fps=source_fps_by_name(cfg),
                    frame_base=int(cfg.get("annotations.frame_base", 0)))
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    lengths = {f.stem: sum(1 for _ in open(f, encoding="utf-8")) - 1 for f in targets.glob("*.csv")}
    fps = _video_fps(cfg)
    main = label_vectors(cfg, cfg.path("paths.annotations"), classifier, lengths, fps)
    other = label_vectors(cfg, second, classifier, lengths, fps)
    both = sorted(set(main) & set(other))
    if not both:
        raise ValueError(f"no video scored by both annotators (second rater: {sorted(other)})")
    try:
        probs = load_probs(cfg, classifier)
    except FileNotFoundError:
        probs = {}
    thr = float(threshold if threshold is not None else (cfg.get("infer.thresholds") or {}).get(classifier, 0.5))
    pen = float(penalty if penalty is not None else (cfg.get("infer.switch_penalty") or {}).get(classifier, 0))
    pairs = {"second rater vs main": [], "model vs main": [], "model vs second rater": []}
    rows = []
    for v in both:
        f = fps.get(v, cfg.fps)
        a, b = main[v], other[v]
        pairs["second rater vs main"].append(compare(a, b, f))
        rows.append({"video": v, "pair": "second rater vs main", **{k: x for k, x in compare(a, b, f).items() if k != "seg"}})
        if v in probs:
            m = viterbi_binary(probs[v]["prob"].to_numpy()[: len(a)], thr, pen)
            for name, ref in (("model vs main", a), ("model vs second rater", b)):
                c = compare(ref, m, f)
                pairs[name].append(c)
                rows.append({"video": v, "pair": name, **{k: x for k, x in c.items() if k != "seg"}})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.simba_project_folder / "logs" / f"agreement_{classifier}_per_video.csv", index=False)
    summary = pd.DataFrame([{"pair": k, "videos": len(v), **_summary(v)} for k, v in pairs.items() if v])
    summary.to_csv(cfg.simba_project_folder / "logs" / f"agreement_{classifier}.csv", index=False)
    LOG.info("%s agreement on %d video(s) (model = held-out leave-one-resident-out predictions):\n%s", classifier, len(both),
             summary.to_string(index=False))
    return summary

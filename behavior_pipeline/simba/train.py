# Kushaan Sharma
"""Train one classifier per behaviour, with optional held-out videos scored afterwards.
Run as the `train` stage of the CLI. algorithm RF uses SimBA's random-forest trainer; ET fits an
extra-trees model here (optionally on a feature subset) and saves it where SimBA's inference finds it.
"""
from __future__ import annotations

import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig
from .project import _ensure_classifiers, set_ini

ENSEMBLE_SECTION = "create ensemble settings"


def _flag(value: object) -> str:
    return "yes" if bool(value) else "no"


def _per_classifier(value: object, classifier: str, default: object) -> object:
    """A train setting may be one value for all behaviours or a {classifier: value} map."""
    if isinstance(value, dict):
        return value.get(classifier, value.get("default", default))
    return default if value is None else value


def ensemble_settings(cfg: PipelineConfig, classifier: str) -> dict[str, object]:
    t = cfg.get("train") or {}
    ev = t.get("evaluation") or {}
    return {
        "pose_estimation_body_parts": cfg.get("simba.pose_estimation_bp_cnt", "16"),
        "model_to_run": "RF",   # the SimBA trainer is only used for RF (see algorithm())
        "classifier": classifier,
        "train_test_size": t.get("train_test_size", 0.2),
        "train_test_split_type": t.get("train_test_split_type", "FRAMES"),
        "under_sample_setting": _per_classifier(t.get("under_sample_setting"), classifier, "None"),
        "under_sample_ratio": _per_classifier(t.get("under_sample_ratio"), classifier, 1.0),
        "over_sample_setting": t.get("over_sample_setting", "None"),
        "over_sample_ratio": t.get("over_sample_ratio", 1.0),
        "rf_n_estimators": t.get("rf_n_estimators", 2000),
        "rf_max_features": t.get("rf_max_features", "sqrt"),
        "rf_criterion": t.get("rf_criterion", "gini"),
        "rf_min_sample_leaf": t.get("rf_min_sample_leaf", 1),
        "rf_max_depth": t.get("rf_max_depth", "None"),
        "class_weights": t.get("class_weights", "None"),
        "class_custom_weights": "None",
        "generate_rf_model_meta_data_file": _flag(ev.get("model_meta_data_file", True)),
        "generate_example_decision_tree": _flag(ev.get("example_decision_tree", False)),
        "generate_example_decision_tree_fancy": "no",
        "generate_classification_report": _flag(ev.get("classification_report", True)),
        "generate_features_importance_log": _flag(ev.get("feature_importance_log", True)),
        "generate_features_importance_bar_graph": _flag(ev.get("feature_importance_bar_graph", True)),
        "n_feature_importance_bars": ev.get("n_feature_importance_bars", 20),
        "compute_feature_permutation_importance": _flag(ev.get("permutation_importance", False)),
        "generate_sklearn_learning_curves": _flag(ev.get("learning_curve", False)),
        "learning_curve_k_splits": 5,
        "learning_curve_data_splits": 5,
        "generate_precision_recall_curves": _flag(ev.get("precision_recall_curve", True)),
        "generate_shap_scores": _flag(ev.get("shap", False)),
        "shap_target_present_no": 100,
        "shap_target_absent_no": 100,
        "shap_save_iteration": "ALL FRAMES",
        "shap_multiprocess": "False",
        "partial_dependency": "no",
        "save_train_test_frm_idx": "no",
        "cuda": "False",
    }


def et_bags(cfg: PipelineConfig, classifier: str) -> int:
    """Number of under-sampled extra-trees bags averaged for this classifier (train.et_bags, 1 = a single model)."""
    return max(int(_per_classifier(cfg.get("train.et_bags"), classifier, 1) or 1), 1)


class UndersampledBagging:
    """Average of n_bags extra-trees models, each fitted on all positives plus its own random draw of
    ratio x as many negatives. Each bag sees different negatives, so the average covers more of the
    rare hard negatives (mounting, allogrooming) than one under-sample does.
    2026-10-09, Attack leave-one-resident-out, 3 bags vs 1: episode F1 0.72 vs 0.70 +- 0.015 (seed spread), frame F1 0.78 either way.
    sklearn-style (fit / predict_proba / classes_), so it sits inside the Pipeline SimBA's inference calls."""

    def __init__(self, n_bags: int = 3, ratio: float | None = 4.0, n_estimators: int = 300, max_features="sqrt",
                 min_samples_leaf: int = 1, class_weight=None, n_jobs: int = -1, random_state: int = 0):
        self.n_bags, self.ratio, self.n_estimators = n_bags, ratio, n_estimators
        self.max_features, self.min_samples_leaf, self.class_weight = max_features, min_samples_leaf, class_weight
        self.n_jobs, self.random_state = n_jobs, random_state

    def get_params(self, deep: bool = True) -> dict:
        return {k: getattr(self, k) for k in ("n_bags", "ratio", "n_estimators", "max_features", "min_samples_leaf",
                                               "class_weight", "n_jobs", "random_state")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier

        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.estimators_ = []
        for b in range(self.n_bags):
            n = neg
            if self.ratio and self.ratio * len(pos) < len(neg):
                n = rng.choice(neg, int(self.ratio * len(pos)), replace=False)
            idx = np.sort(np.concatenate([pos, n]))
            et = ExtraTreesClassifier(n_estimators=self.n_estimators, max_features=self.max_features,
                                      min_samples_leaf=self.min_samples_leaf, class_weight=self.class_weight,
                                      n_jobs=self.n_jobs, random_state=self.random_state + b)
            self.estimators_.append(et.fit(X[idx], y[idx]))
        self.classes_ = self.estimators_[0].classes_
        self.n_features_in_ = self.estimators_[0].n_features_in_
        self.feature_importances_ = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        return self

    def predict_proba(self, X):
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def algorithm(cfg: PipelineConfig, classifier: str) -> str:
    return str(_per_classifier(cfg.get("train.algorithm"), classifier, "RF")).upper()


def kept_features(cfg: PipelineConfig, features: list[str], classifier: str = "") -> list[int]:
    """Indices of the features the model sees (ET only): train.keep_features_re keeps matching names,
    then train.drop_features_re removes matching names. Either may be a {classifier: pattern} map."""
    keep_pat = _per_classifier(cfg.get("train.keep_features_re"), classifier, None)
    drop_pat = _per_classifier(cfg.get("train.drop_features_re"), classifier, None)
    idx = list(range(len(features)))
    if keep_pat:
        rx = re.compile(str(keep_pat))
        idx = [i for i in idx if rx.search(features[i])]
    if drop_pat:
        rx = re.compile(str(drop_pat))
        idx = [i for i in idx if not rx.search(features[i])]
    return idx


def make_estimator(cfg: PipelineConfig, classifier: str, features: list[str]):
    """Unfitted extra-trees pipeline that takes every SimBA feature column and keeps its own subset.

    The column selection lives inside the model, so SimBA's inference can pass the full feature table.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import Pipeline

    t = cfg.get("train") or {}
    cw = str(t.get("class_weights", "None"))
    kw = dict(n_estimators=int(t.get("et_n_estimators", 300)), max_features=t.get("rf_max_features", "sqrt"),
              min_samples_leaf=int(t.get("rf_min_sample_leaf", 1)), class_weight=None if cw.lower() == "none" else cw,
              n_jobs=int(t.get("n_jobs", -1)), random_state=int(t.get("seed", 0)))
    bags = et_bags(cfg, classifier)
    if bags > 1:
        # the bags under-sample for themselves; undersample() then passes every row through
        s = ensemble_settings(cfg, classifier)
        ratio = float(s["under_sample_ratio"]) if str(s["under_sample_setting"]).lower() == "random undersample" else None
        trees = UndersampledBagging(n_bags=bags, ratio=ratio, **kw)
    else:
        trees = ExtraTreesClassifier(**kw)
    keep = kept_features(cfg, features, classifier)
    if not keep:
        raise ValueError(f"{classifier}: train.keep_features_re / drop_features_re leave no feature columns "
                         f"(did the features-extra stage run?)")
    select = ColumnTransformer([("keep", "passthrough", keep)], remainder="drop")
    model = Pipeline([("select", select), ("trees", trees)])
    model.feature_names_all = list(features)
    model.feature_names_kept = [features[i] for i in keep]
    return model


def undersample(cfg: PipelineConfig, classifier: str, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices after random under-sampling of the negatives (all rows when switched off or short of negatives)."""
    s = ensemble_settings(cfg, classifier)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if algorithm(cfg, classifier) == "ET" and et_bags(cfg, classifier) > 1:
        return np.arange(len(y))     # UndersampledBagging draws its own negatives per bag
    if str(s["under_sample_setting"]).lower() == "random undersample":
        n = int(float(s["under_sample_ratio"]) * len(pos))
        if n < len(neg):
            neg = rng.choice(neg, n, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def feature_columns(cfg: PipelineConfig, columns: list[str]) -> list[str]:
    """The columns SimBA feeds a model: everything except pose coordinates and target columns."""
    from .evaluate import pose_columns

    drop = set(pose_columns(cfg)) | set(_target_columns(cfg)) | set(cfg.classifiers)
    return [c for c in columns if c not in drop]


def train_extra_trees(cfg: PipelineConfig, classifier: str, n_estimators: int | None = None) -> Path:
    """Fit the extra-trees pipeline on targets_inserted (already the masked view) and save <clf>.sav."""
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    frames = [pd.read_csv(f, index_col=0) for f in sorted(targets.glob("*.csv"))]
    frames = [f for f in frames if classifier in f.columns]
    df = pd.concat(frames, ignore_index=True)
    features = feature_columns(cfg, list(frames[0].columns))
    X = df[features].fillna(0).to_numpy(np.float32)
    y = df[classifier].to_numpy().astype(int)
    idx = undersample(cfg, classifier, y, np.random.default_rng(int(cfg.get("train.seed", 0))))
    model = make_estimator(cfg, classifier, features)
    if n_estimators is not None:
        model.set_params(trees__n_estimators=n_estimators)
    LOG.info("training %s (ET, %d trees x %d bag(s), %d of %d features, %d frames, %d positive)", classifier,
             model.named_steps["trees"].n_estimators, et_bags(cfg, classifier), len(model.feature_names_kept), len(features),
             len(idx), int(y[idx].sum()))
    model.fit(X[idx], y[idx])
    out = cfg.simba_models_dir / f"{classifier}.sav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(model, fh)
    imp = pd.DataFrame({"FEATURE": model.feature_names_kept, "FEATURE_IMPORTANCE": model.named_steps["trees"].feature_importances_})
    evals = cfg.simba_models_dir / "model_evaluations"
    evals.mkdir(parents=True, exist_ok=True)
    imp.sort_values("FEATURE_IMPORTANCE", ascending=False).to_csv(evals / f"{classifier}_feature_importance_log.csv", index=False)
    LOG.info("saved %s", out)
    return out


def _holdout_paths(cfg: PipelineConfig) -> tuple[Path, Path]:
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    holdout = cfg.simba_project_folder / "csv" / "targets_holdout"
    return targets, holdout


def _move_holdout(cfg: PipelineConfig, names: list[str]) -> list[Path]:
    targets, holdout = _holdout_paths(cfg)
    holdout.mkdir(parents=True, exist_ok=True)
    moved = []
    for n in names:
        src = targets / f"{n}.csv"
        if src.exists():
            shutil.move(str(src), holdout / src.name)
            moved.append(holdout / src.name)
        else:
            LOG.warning("holdout video %s has no targets file", n)
    return moved


def _restore_holdout(cfg: PipelineConfig) -> None:
    targets, holdout = _holdout_paths(cfg)
    if holdout.exists():
        for f in holdout.glob("*.csv"):
            shutil.move(str(f), targets / f.name)


def _mask_paths(cfg: PipelineConfig) -> tuple[Path, Path]:
    csv = cfg.simba_project_folder / "csv"
    return csv / "targets_masked", csv / "targets_unmasked"


def _restore_masked(cfg: PipelineConfig) -> None:
    """Put the full targets_inserted back if a masked view is (still) in its place."""
    targets, _ = _holdout_paths(cfg)
    masked, aside = _mask_paths(cfg)
    if aside.exists():
        if targets.exists():
            shutil.rmtree(targets)
        shutil.move(str(aside), str(targets))
    if masked.exists():
        shutil.rmtree(masked)


def _masked_targets_view(cfg: PipelineConfig, classifier: str) -> pd.DataFrame:
    """Swap targets_inserted for a copy without the frames whose label for `classifier` is unknown.

    SimBA's trainer concatenates the target files and never needs consecutive frames, so dropping
    rows is enough. Returns the per-video kept/dropped table; call _restore_masked afterwards.
    """
    from .annotations import read_mask

    targets, _ = _holdout_paths(cfg)
    masked, aside = _mask_paths(cfg)
    _restore_masked(cfg)
    masked.mkdir(parents=True)
    rows = []
    for f in sorted(targets.glob("*.csv")):
        m = read_mask(cfg, f.stem, classifier)
        if m is None:
            shutil.copy(str(f), masked / f.name)
            rows.append({"classifier": classifier, "video": f.stem, "kept": None, "dropped": 0, "note": "no mask, all frames used"})
            continue
        df = pd.read_csv(f, index_col=0)
        keep = m.to_numpy().astype(bool)[: len(df)]
        if len(keep) < len(df):
            keep = list(keep) + [True] * (len(df) - len(keep))
        keep = pd.Series(keep, index=df.index)
        kept = int(keep.sum())
        if kept:
            df[keep].to_csv(masked / f.name)
        rows.append({"classifier": classifier, "video": f.stem, "kept": kept, "dropped": int(len(df) - kept),
                     "positives": int(df.loc[keep, classifier].sum()) if classifier in df.columns else 0,
                     "note": "" if kept else "no known frames, left out"})
    shutil.move(str(targets), str(aside))
    shutil.move(str(masked), str(targets))
    table = pd.DataFrame(rows)
    LOG.info("%s: %d frame(s) with unknown labels dropped from training", classifier, int(table["dropped"].sum()))
    return table


def train_classifier(cfg: PipelineConfig, classifier: str, n_estimators: int | None = None,
                     positives: int | None = None, negatives: int | None = None) -> Path:
    if algorithm(cfg, classifier) == "ET":
        return train_extra_trees(cfg, classifier, n_estimators)
    from simba.model.train_rf import TrainRandomForestClassifier

    settings = ensemble_settings(cfg, classifier)
    if n_estimators is not None:
        settings["rf_n_estimators"] = n_estimators
    ratio = float(settings["under_sample_ratio"] or 0)
    if str(settings["under_sample_setting"]).lower() == "random undersample" and positives and negatives is not None:
        # SimBA refuses a ratio the data cannot supply; fall back to every negative frame
        if ratio * positives > negatives:
            LOG.warning("%s: %.1f x %d positives exceeds the %d negatives; using all negatives (no under-sampling)",
                        classifier, ratio, positives, negatives)
            settings["under_sample_setting"] = "None"
    set_ini(cfg, ENSEMBLE_SECTION, settings)
    LOG.info("training %s (RF, %s trees)", classifier, settings["rf_n_estimators"])
    trainer = TrainRandomForestClassifier(config_path=str(cfg.simba_config_path))
    trainer.run()
    trainer.save()
    model = cfg.simba_models_dir / f"{classifier}.sav"
    if not model.exists():
        raise RuntimeError(f"SimBA did not write {model}")
    LOG.info("saved %s", model)
    return model


def score_holdout(cfg: PipelineConfig, models: dict[str, Path], threshold: float = 0.5) -> pd.DataFrame | None:
    """Score held-out videos with the new models; precision/recall/F1 per classifier."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    _, holdout = _holdout_paths(cfg)
    files = sorted(holdout.glob("*.csv")) if holdout.exists() else []
    if not files:
        return None
    rows = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        for clf, model_path in models.items():
            if clf not in df.columns:
                continue
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)
            X = df.drop(columns=[c for c in cfg.classifiers if c in df.columns])
            if hasattr(model, "feature_names_all"):
                X = X[model.feature_names_all].fillna(0).to_numpy(np.float32)
            elif hasattr(model, "feature_names_in_"):
                X = X[[c for c in model.feature_names_in_ if c in X.columns]]
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= threshold).astype(int)
            y = df[clf].astype(int)
            rows.append({"video": f.stem, "classifier": clf, "frames": len(df), "positives": int(y.sum()),
                         "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
                         "f1": f1_score(y, pred, zero_division=0)})
    table = pd.DataFrame(rows)
    table.to_csv(cfg.simba_project_folder / "logs" / "holdout_validation.csv", index=False)
    LOG.info("holdout validation:\n%s", table.to_string(index=False))
    return table


def _target_columns(cfg: PipelineConfig) -> list[str]:
    """Columns in a targets file that are not features: the classifiers SimBA must know about."""
    targets = cfg.simba_project_folder / "csv" / "targets_inserted"
    features = cfg.simba_project_folder / "csv" / "features_extracted"
    t = next(iter(sorted(targets.glob("*.csv"))), None)
    if t is None or not (features / t.name).exists():
        return []
    feat_cols = set(pd.read_csv(features / t.name, nrows=0).columns)
    return [c for c in pd.read_csv(t, nrows=0).columns if c not in feat_cols]


def train_all(cfg: PipelineConfig, classifiers: list[str] | None = None, n_estimators: int | None = None) -> dict[str, Path]:
    classifiers = classifiers or cfg.classifiers
    holdout = list(cfg.get("train.holdout_videos") or [])
    # Restore first in case an earlier run stopped with files still moved aside.
    _restore_masked(cfg)
    _restore_holdout(cfg)
    _ensure_classifiers(cfg, extra=_target_columns(cfg))
    if holdout:
        _move_holdout(cfg, holdout)
    models: dict[str, Path] = {}
    mask_tables = []
    try:
        for clf in classifiers:
            table = _masked_targets_view(cfg, clf)
            mask_tables.append(table)
            try:
                pos = int(table["positives"].fillna(0).sum()) if not table.empty else 0
                if pos == 0:
                    LOG.warning("%s: no labelled frames, not trained", clf)
                    continue
                kept = int(table["kept"].fillna(0).sum())
                models[clf] = train_classifier(cfg, clf, n_estimators=n_estimators, positives=pos, negatives=max(kept - pos, 0))
            finally:
                _restore_masked(cfg)
        if holdout:
            score_holdout(cfg, models)
    finally:
        _restore_masked(cfg)
        _restore_holdout(cfg)
        if mask_tables:
            pd.concat(mask_tables, ignore_index=True).to_csv(cfg.simba_project_folder / "logs" / "label_masks_summary.csv", index=False)
    return models

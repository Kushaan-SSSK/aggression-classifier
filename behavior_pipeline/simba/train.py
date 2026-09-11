# Kushaan Sharma
"""Train one SimBA random-forest classifier per behaviour, with optional held-out
videos scored afterwards. Run as the `train` stage of the CLI.
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pandas as pd

from ..config import LOG, PipelineConfig
from .project import set_ini

ENSEMBLE_SECTION = "create ensemble settings"


def _flag(value: object) -> str:
    return "yes" if bool(value) else "no"


def ensemble_settings(cfg: PipelineConfig, classifier: str) -> dict[str, object]:
    t = cfg.get("train") or {}
    ev = t.get("evaluation") or {}
    return {
        "pose_estimation_body_parts": cfg.get("simba.pose_estimation_bp_cnt", "16"),
        "model_to_run": str(t.get("algorithm", "RF")),
        "classifier": classifier,
        "train_test_size": t.get("train_test_size", 0.2),
        "train_test_split_type": t.get("train_test_split_type", "FRAMES"),
        "under_sample_setting": t.get("under_sample_setting", "None"),
        "under_sample_ratio": t.get("under_sample_ratio", 1.0),
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


def train_classifier(cfg: PipelineConfig, classifier: str, n_estimators: int | None = None) -> Path:
    from simba.model.train_rf import TrainRandomForestClassifier

    settings = ensemble_settings(cfg, classifier)
    if n_estimators is not None:
        settings["rf_n_estimators"] = n_estimators
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
            if hasattr(model, "feature_names_in_"):
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


def train_all(cfg: PipelineConfig, classifiers: list[str] | None = None, n_estimators: int | None = None) -> dict[str, Path]:
    classifiers = classifiers or cfg.classifiers
    holdout = list(cfg.get("train.holdout_videos") or [])
    # Restore first in case an earlier run stopped with files still moved aside.
    _restore_holdout(cfg)
    if holdout:
        _move_holdout(cfg, holdout)
    models: dict[str, Path] = {}
    try:
        for clf in classifiers:
            models[clf] = train_classifier(cfg, clf, n_estimators=n_estimators)
        if holdout:
            score_holdout(cfg, models)
    finally:
        _restore_holdout(cfg)
    return models

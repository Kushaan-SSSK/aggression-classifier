# Kushaan Sharma
"""SimBA unsupervised motif discovery: bout dataset, UMAP grid, HDBSCAN grid and
per-cluster statistics. Run as the `unsupervised` stage of the CLI.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..config import LOG, PipelineConfig


def _alias_model_names(obj) -> None:
    """Patch attributes that UmapEmbedder / HDBSCANClusterer.fit() read but never set (SimBA bug)."""
    if not hasattr(obj, "timer"):
        from simba.utils.printing import SimbaTimer

        obj.timer = SimbaTimer(start=True)
    if not hasattr(obj, "model_names"):
        names = getattr(obj, "mdl_names", None)
        if not names:
            try:
                from simba.utils.lookups import get_model_names

                names = get_model_names()
            except Exception:
                names = [f"model_{i}" for i in range(1000)]
        obj.model_names = list(names)


def _settings(cfg: PipelineConfig) -> dict:
    from simba.unsupervised.enums import Unsupervised as U

    u = cfg.get("unsupervised") or {}
    return {
        U.DATA_SLICE_SELECTION.value: str(u.get("data_slice", U.ALL_FEATURES_EX_POSE.value)),
        U.CLF_SLICE_SELECTION.value: str(u.get("clf_slice", "ALL CLASSIFIERS")),
        U.BOUT_AGGREGATION_TYPE.value: str(u.get("bout_aggregation_type", "MEDIAN")),
        U.MIN_BOUT_LENGTH.value: int(u.get("min_bout_length", 5)),
        U.FEATURE_PATH.value: None,
    }


def create_dataset(cfg: PipelineConfig) -> Path:
    from simba.unsupervised.dataset_creator import DatasetCreator

    logs = cfg.simba_project_folder / "logs"
    before = set(logs.glob("unsupervised_data_*.pickle"))
    settings = _settings(cfg)
    from simba.unsupervised.enums import Unsupervised as U

    # SimBA only slices on names registered as targets, so an injected column such as
    # Proximity has to be added to project_config.ini first.
    clf_slice = settings[U.CLF_SLICE_SELECTION.value]
    if clf_slice and not clf_slice.upper().startswith("ALL") and clf_slice not in cfg.classifiers:
        from .project import _ensure_classifiers

        _ensure_classifiers(cfg, extra=[clf_slice])
    LOG.info("unsupervised dataset: %s", settings)
    creator = DatasetCreator(config_path=str(cfg.simba_config_path), settings=settings)
    creator.run()
    new = sorted(set(logs.glob("unsupervised_data_*.pickle")) - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError("DatasetCreator produced no pickle in project_folder/logs")
    LOG.info("dataset -> %s", new[-1].name)
    return new[-1]


def embed(cfg: PipelineConfig, data_path: Path, out_dir: Path) -> Path:
    from simba.unsupervised.umap_embedder import UmapEmbedder

    u = cfg.get("unsupervised.umap") or {}
    hp = {"n_neighbors": list(u.get("n_neighbors", [15])), "min_distance": list(u.get("min_distance", [0.1])),
          "spread": list(u.get("spread", [1.0])), "scaler": str(u.get("scaler", "MIN-MAX")), "variance": float(u.get("variance", 0.25))}
    save_dir = out_dir / "umap"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    LOG.info("UMAP grid %s", hp)
    embedder = UmapEmbedder()
    _alias_model_names(embedder)
    embedder.fit(data_path=str(data_path), save_dir=str(save_dir), hyper_parameters=hp)
    if not list(save_dir.glob("*.pickle")):
        raise RuntimeError(f"UMAP produced no models in {save_dir}")
    return save_dir


def cluster(cfg: PipelineConfig, umap_dir: Path, out_dir: Path) -> Path:
    from simba.unsupervised.hdbscan_clusterer import HDBSCANClusterer

    h = cfg.get("unsupervised.hdbscan") or {}
    hp = {"alpha": list(h.get("alpha", [1.0])), "min_cluster_size": list(h.get("min_cluster_size", [15])),
          "min_samples": list(h.get("min_samples", [1])), "cluster_selection_epsilon": list(h.get("cluster_selection_epsilon", [0.0]))}
    save_dir = out_dir / "hdbscan"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    LOG.info("HDBSCAN grid %s", hp)
    clusterer = HDBSCANClusterer()
    _alias_model_names(clusterer)
    clusterer.fit(data_path=str(umap_dir), save_dir=str(save_dir), hyper_parameters=hp)
    if not list(save_dir.glob("*.pickle")):
        raise RuntimeError(f"HDBSCAN produced no models in {save_dir}")
    return save_dir


def describe_clusters(cfg: PipelineConfig, hdbscan_dir: Path) -> list[Path]:
    """Cluster statistics with SimBA's calculator; failures are logged, not fatal."""
    from simba.unsupervised.cluster_frequentist_calculator import ClusterFrequentistCalculator

    done: list[Path] = []
    settings = {"scaled": True, "anova": True, "tukey_posthoc": False, "descriptive_statistics": True, "kruskal_wallis": False}
    for pickle_path in sorted(hdbscan_dir.glob("*.pickle")):
        try:
            calc = ClusterFrequentistCalculator(config_path=str(cfg.simba_config_path), data_path=str(pickle_path), settings=settings)
            calc.run()
            done.append(pickle_path)
        except Exception as exc:
            LOG.warning("cluster statistics for %s failed: %s", pickle_path.name, exc)
    return done


def run_unsupervised(cfg: PipelineConfig, out_dir: Path | None = None, with_stats: bool = True) -> Path:
    out_dir = out_dir or (cfg.simba_project_folder / "unsupervised")
    out_dir.mkdir(parents=True, exist_ok=True)
    data = create_dataset(cfg)
    umap_dir = embed(cfg, data, out_dir)
    hdb_dir = cluster(cfg, umap_dir, out_dir)
    if with_stats:
        describe_clusters(cfg, hdb_dir)
    LOG.info("unsupervised outputs in %s", out_dir)
    return out_dir

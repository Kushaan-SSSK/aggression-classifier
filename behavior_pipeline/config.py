# Kushaan Sharma

"""Load config/pipeline.yaml and resolve paths relative to the project root."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("behavior_pipeline")

DEFAULT_CONFIG_NAME = os.path.join("config", "pipeline.yaml")

# SimBA's "2 animals; 16 body-parts" column order; do not reorder
SIMBA_16BP_PER_ANIMAL = ["Ear_left", "Ear_right", "Nose", "Center", "Lat_left", "Lat_right", "Tail_base", "Tail_end"]
SIMBA_16BP_COLUMNS = [f"{bp}_{i}" for i in (1, 2) for bp in SIMBA_16BP_PER_ANIMAL]


def setup_logging(level: int = logging.INFO) -> None:
    if not LOG.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        LOG.addHandler(handler)
    LOG.setLevel(level)


@dataclass
class PipelineConfig:
    """The YAML dict with dotted-key access and path resolution."""

    raw: dict[str, Any]
    root: Path
    config_path: Path
    _overrides: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path(self, dotted: str, default: str | None = None) -> Path:
        """Return the value at ``dotted`` as a path, relative ones resolved against the root."""
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(f"config key {dotted!r} is not set")
        p = Path(str(value))
        return p if p.is_absolute() else (self.root / p)

    @property
    def simba_project_dir(self) -> Path:
        return self.path("project.simba_project_dir")

    @property
    def simba_project_folder(self) -> Path:
        """The folder SimBA calls project_folder, holding project_config.ini."""
        return self.simba_project_dir / self.get("project.simba_project_name") / "project_folder"

    @property
    def simba_config_path(self) -> Path:
        return self.simba_project_folder / "project_config.ini"

    @property
    def simba_models_dir(self) -> Path:
        """Where SimBA saves trained classifiers, a sibling of project_folder."""
        return self.simba_project_folder.parent / "models" / "generated_models"

    @property
    def classifiers(self) -> list[str]:
        return list(self.get("project.classifiers", []))

    @property
    def fps(self) -> float:
        return float(self.get("video.fps", 15))


def load_config(config_path: str | os.PathLike | None = None, root: str | os.PathLike | None = None) -> PipelineConfig:
    """Load the YAML config; the root defaults to the parent of the config folder."""
    if config_path is None:
        config_path = Path.cwd() / DEFAULT_CONFIG_NAME
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    root_path = Path(root).resolve() if root is not None else config_path.parent.parent
    return PipelineConfig(raw=raw, root=root_path, config_path=config_path)

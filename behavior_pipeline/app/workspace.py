# Kushaan Sharma
"""An analysis workspace for one folder of new recordings, kept apart from the training project.

The workspace (default <video folder>/aggression_analysis) holds its own copy of config/pipeline.yaml, so every
relative path in it (processed videos, pose, SimBA project, results) resolves inside the workspace, and a copy of
each trained model. The stages run as `python -m behavior_pipeline <stage> --config <workspace config>`, so the app
does exactly what the command line does.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ..config import PipelineConfig, load_config
from ..video.manifest import safe_video_name

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_CONFIG = REPO / "config" / "pipeline.yaml"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
WORKSPACE_DIRNAME = "aggression_analysis"
MANIFEST_COLUMNS = ["video_name", "source_path", "intruder_entry_s", "crop_x", "crop_y", "crop_w", "crop_h", "downsample",
                    "cage_width_mm", "clip_length_s", "group", "cohort", "lighting", "notes"]


def venv_python(name: str) -> str:
    """The interpreter of one of the repo's environments (.venv-simba / .venv-dlc), else this one."""
    for cand in (REPO / name / "Scripts" / "python.exe", REPO / name / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return sys.executable


@dataclass
class VideoSetup:
    """One recording and what preprocessing needs to know about it."""

    source: str
    name: str
    entry_s: float | None = None                       # intruder entry, seconds into the recording
    crop: tuple[int, int, int, int] | None = None      # x, y, w, h in recording pixels
    group: str = ""
    fps: float = 0.0
    frames: int = 0
    width: int = 0
    height: int = 0
    note: str = ""

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def ready(self) -> bool:
        return self.entry_s is not None and self.crop is not None


def probe(path: str | Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return {"fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)}
    finally:
        cap.release()


def scan_folder(folder: str | Path, recursive: bool = False) -> list[Path]:
    """Video files in a folder, skipping anything inside an analysis workspace."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXT and WORKSPACE_DIRNAME not in p.parts)


def batch_log_settings(folder: str | Path) -> dict[str, tuple[float, tuple[int, int, int, int] | None]]:
    """Recording stem -> (intruder entry s, crop) from a SimBA batch_process_log.json in the folder or its parent."""
    from ..video.adopt import _hms_to_s, load_batch_log

    folder = Path(folder)
    for cand in (folder / "batch_process_log.json", folder.parent / "batch_process_log.json"):
        if cand.exists():
            out = {}
            for stem, e in load_batch_log(cand).items():
                c = e.get("crop_settings") or {}
                crop = (int(c["top_left_x"]), int(c["top_left_y"]), int(c["width"]), int(c["height"])) if e.get("crop") and c else None
                out[stem] = (_hms_to_s((e.get("clip_settings") or {}).get("start", "0")) if e.get("clip") else 0.0, crop)
            return out
    return {}


def unique_names(paths: list[Path]) -> list[str]:
    names, seen = [], set()
    for p in paths:
        base = safe_video_name(p.stem) or "video"
        n, k = base, 2
        while n in seen:
            n, k = f"{base}_{k}", k + 1
        seen.add(n)
        names.append(n)
    return names


def model_files(clf: str, template: Path = TEMPLATE_CONFIG) -> tuple[Path | None, Path | None]:
    """(model .sav, model card .json) for one behaviour: the training project's own model if it has one, else the
    downloaded copy in <repo>/models (env/get_model.py puts the released model there)."""
    cfg = load_config(template)
    for sav, card in ((cfg.simba_models_dir / f"{clf}.sav", cfg.simba_project_folder / "logs" / f"{clf}_model_card.json"),
                      (REPO / "models" / f"{clf}.sav", REPO / "models" / f"{clf}_model_card.json")):
        if sav.exists():
            return sav, (card if card.exists() else None)
    return None, None


def available_models(template: Path = TEMPLATE_CONFIG) -> dict[str, dict]:
    """Trained models the app can use: behaviour -> {path, trained, card}."""
    out = {}
    for clf in load_config(template).classifiers:
        sav, card_path = model_files(clf, template)
        if sav is None:
            continue
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path else None
        trained = (card or {}).get("model_trained") or time.strftime("%Y-%m-%d", time.localtime(sav.stat().st_mtime))
        out[clf] = {"path": sav, "trained": trained[:10], "card": card}
    return out


@dataclass
class Stage:
    key: str
    title: str
    python: str
    args: list[str] = field(default_factory=list)


class Workspace:
    def __init__(self, root: str | Path, template: Path = TEMPLATE_CONFIG):
        self.root = Path(root)
        self.template = Path(template)

    @classmethod
    def for_folder(cls, video_folder: str | Path) -> "Workspace":
        return cls(Path(video_folder) / WORKSPACE_DIRNAME)

    @property
    def config_path(self) -> Path:
        return self.root / "config" / "pipeline.yaml"

    @property
    def state_path(self) -> Path:
        return self.root / "workspace.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def exists(self) -> bool:
        return self.config_path.exists()

    def cfg(self) -> PipelineConfig:
        return load_config(self.config_path)

    def state(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}

    def create(self, classifiers: list[str], video_folder: str | Path) -> None:
        """Write the workspace config (the template's settings, workspace-relative paths) and copy the models in.
        Re-running refreshes both, so a retrained model is picked up."""
        raw = yaml.safe_load(self.template.read_text(encoding="utf-8")) or {}
        raw.setdefault("project", {})["classifiers"] = list(classifiers)
        # models outside the SimBA tree: SimBA will not create a project in a folder that already exists
        raw.setdefault("paths", {})["trained_models"] = "models"
        # short SimBA folder names: Windows refuses paths over 260 characters unless long paths are enabled
        raw["project"]["simba_project_dir"] = "simba"
        raw["project"]["simba_project_name"] = "ri"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Made by the aggression-classifier app from {self.template} on {time.strftime('%Y-%m-%d %H:%M')}.\n"
                  "# Paths are relative to this workspace. Edit the template, not this copy: the app rewrites it.\n")
        self.config_path.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        cfg = self.cfg()
        cfg.simba_models_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "model_cards").mkdir(exist_ok=True)
        models = {}
        for clf in classifiers:
            sav, card = model_files(clf, self.template)
            if sav is None:
                raise FileNotFoundError(f"no trained {clf} model; run env/get_model.py")
            shutil.copy2(sav, cfg.simba_models_dir / sav.name)
            if card is not None:
                shutil.copy2(card, self.root / "model_cards" / card.name)
            models[clf] = time.strftime("%Y-%m-%d %H:%M", time.localtime(sav.stat().st_mtime))
        self.state_path.write_text(json.dumps({"video_folder": str(video_folder), "template": str(self.template),
                                               "classifiers": list(classifiers), "models_trained": models,
                                               "updated": time.strftime("%Y-%m-%d %H:%M")}, indent=2), encoding="utf-8")

    def longest_path(self, names: list[str]) -> int:
        """Length of the deepest file the pipeline will write (SimBA's outlier folder), for the 260-character check."""
        deepest = Path("simba") / "ri" / "project_folder" / "csv" / "outlier_corrected_movement_location"
        return len(str(self.root.resolve() / deepest)) + 1 + max((len(n) for n in names), default=0) + len(".csv")

    # ---- manifest ---------------------------------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / "data" / "annotations" / "video_manifest.csv"

    def write_manifest(self, setups: list[VideoSetup]) -> Path:
        rows = []
        for s in setups:
            if not s.ready:
                continue
            x, y, w, h = s.crop
            rows.append({"video_name": s.name, "source_path": str(Path(s.source).resolve()), "intruder_entry_s": round(float(s.entry_s), 3),
                         "crop_x": x, "crop_y": y, "crop_w": w, "crop_h": h, "group": s.group,
                         "notes": f"set in the app; source {s.width}x{s.height} @ {s.fps:.0f} fps"})
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(self.manifest_path, index=False)
        return self.manifest_path

    def read_manifest(self) -> dict[str, VideoSetup]:
        """Saved setups by source path, so reopening a folder keeps the crops and entry times."""
        if not self.manifest_path.exists():
            return {}
        df = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        out = {}
        for _, r in df.iterrows():
            crop = None
            if all(str(r.get(c, "")).strip() for c in ("crop_x", "crop_y", "crop_w", "crop_h")):
                crop = tuple(int(float(r[c])) for c in ("crop_x", "crop_y", "crop_w", "crop_h"))
            entry = float(r["intruder_entry_s"]) if str(r.get("intruder_entry_s", "")).strip() else None
            out[str(Path(r["source_path"]).resolve())] = VideoSetup(source=r["source_path"], name=r["video_name"], entry_s=entry,
                                                                     crop=crop, group=str(r.get("group", "")))
        return out

    def save_setups(self, setups: list[VideoSetup]) -> None:
        """Everything the table holds, ready or not, so nothing typed is lost between sessions."""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "setups.json").write_text(json.dumps([asdict(s) for s in setups], indent=1), encoding="utf-8")

    def load_setups(self) -> dict[str, VideoSetup]:
        p = self.root / "setups.json"
        if not p.exists():
            return self.read_manifest()
        out = {}
        for d in json.loads(p.read_text(encoding="utf-8")):
            d["crop"] = tuple(d["crop"]) if d.get("crop") else None
            out[str(Path(d["source"]).resolve())] = VideoSetup(**d)
        return out

    def processed(self, name: str) -> dict | None:
        """The preprocessing sidecar of a video, if it was processed in this workspace."""
        p = self.root / "data" / "processed_videos" / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def changed(self, s: VideoSetup) -> bool:
        """True when a processed video was made with a different crop or entry time than the setup now says."""
        meta = self.processed(s.name)
        if meta is None or not s.ready:
            return False
        crop = tuple(meta["crop"]) if meta.get("crop") else None
        return crop != tuple(s.crop) or abs(float(meta.get("intruder_entry_s", -1)) - float(s.entry_s)) > 1e-3

    def forget(self, name: str) -> list[Path]:
        """Delete everything derived from one video (exact file names only), so the next run redoes it from scratch.
        Pose files are left: they go stale against the new processed video and are redone by pose --missing."""
        proj = self.root / "simba"
        cands = [self.root / "data" / "processed_videos", self.root / "data" / "resident_clips", self.root / "data" / "pose" / "simba_16bp"]
        cands += [d for d in proj.glob("*/project_folder/csv/*") if d.is_dir()] + list(proj.glob("*/project_folder/videos"))
        gone = []
        for d in cands:
            for f in d.glob(f"{name}.*"):
                if f.stem == name:
                    f.unlink()
                    gone.append(f)
        for f in (self.root / "data" / "pose" / "resident").glob(f"{name}.resident*"):
            f.unlink()
            gone.append(f)
        return gone

    # ---- stages -----------------------------------------------------------------------------------------------
    def stages(self, classifiers: list[str]) -> list[Stage]:
        simba, dlc, c = venv_python(".venv-simba"), venv_python(".venv-dlc"), ["--config", str(self.config_path)]
        return [Stage("preprocess", "Preparing videos", simba, ["preprocess", *c]),
                Stage("pose", "Tracking the mice", dlc, ["pose", "--missing", *c]),
                Stage("resident", "Identifying the resident", simba, ["resident", *c]),
                Stage("convert", "Converting tracking", simba, ["convert", *c]),
                Stage("project", "Building features", simba, ["project", *c]),
                Stage("infer", "Scoring behaviour", simba, ["infer", "--clf", *classifiers, *c]),
                Stage("report", "Summarising", simba, ["report", "--clf", *classifiers, *c])]

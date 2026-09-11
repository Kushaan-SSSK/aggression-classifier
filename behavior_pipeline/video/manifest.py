# Kushaan Sharma

"""Read and validate the two hand-filled CSVs: the video manifest and the annotations.

Video manifest columns (config/video_manifest.template.csv):
    video_name        unique id, used as the file stem downstream (letters, digits, _ and -)
    source_path       raw video file, absolute or relative to the project root
    intruder_entry_s  seconds into the raw video when the intruder is placed in the cage
    crop_x, crop_y, crop_w, crop_h   crop rectangle in raw-video pixels, blank for no crop
    downsample        blank or 0 keeps the resolution, otherwise the target width in px
    cage_width_mm     inside cage width spanned by crop_w, blank uses simba.cage_width_mm
    cohort, lighting, notes   free text

Annotation columns (config/annotations.template.csv):
    video_name, behavior, start_s, end_s, annotator, notes
    Times are seconds in the processed clip, so 0 is the intruder entry.
    behavior == NONE with blank times marks a fully scored video with no events.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import PipelineConfig

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class ManifestError(ValueError):
    pass


@dataclass
class VideoEntry:
    video_name: str
    source_path: Path
    intruder_entry_s: float
    crop: tuple[int, int, int, int] | None
    downsample_width: int | None
    cage_width_mm: float | None
    cohort: str = ""
    lighting: str = ""
    notes: str = ""


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _to_float(value, col: str, row: int, required: bool = False) -> float | None:
    if _blank(value):
        if required:
            raise ManifestError(f"row {row}: column {col!r} is required")
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"row {row}: column {col!r} must be a number, got {value!r}") from exc


def read_video_manifest(path: str | Path, cfg: PipelineConfig | None = None) -> list[VideoEntry]:
    """Read and validate the video manifest CSV."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"video manifest not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"video_name", "source_path", "intruder_entry_s"}
    missing = required - set(df.columns)
    if missing:
        raise ManifestError(f"video manifest is missing columns: {sorted(missing)}")

    root = cfg.root if cfg is not None else path.parent
    entries: list[VideoEntry] = []
    seen: set[str] = set()
    for i, row in df.iterrows():
        # spreadsheet row number: 1-based plus the header line
        rown = int(i) + 2
        name = str(row["video_name"]).strip()
        if not name:
            continue
        if not _NAME_RE.match(name):
            raise ManifestError(f"row {rown}: video_name {name!r} may only contain letters, digits, _ and -")
        if name in seen:
            raise ManifestError(f"row {rown}: duplicate video_name {name!r}")
        seen.add(name)
        src = Path(str(row["source_path"]).strip())
        if not src.is_absolute():
            src = root / src
        entry_s = _to_float(row["intruder_entry_s"], "intruder_entry_s", rown, required=True)
        crop_vals = [row.get(c, "") for c in ("crop_x", "crop_y", "crop_w", "crop_h")]
        if all(_blank(v) for v in crop_vals):
            crop = None
        elif any(_blank(v) for v in crop_vals):
            raise ManifestError(f"row {rown}: give all four of crop_x, crop_y, crop_w, crop_h or none")
        else:
            crop = tuple(int(float(v)) for v in crop_vals)
            if crop[2] <= 0 or crop[3] <= 0:
                raise ManifestError(f"row {rown}: crop_w and crop_h must be positive")
        ds = _to_float(row.get("downsample", ""), "downsample", rown)
        ds_width = int(ds) if ds and ds > 0 else None
        cage = _to_float(row.get("cage_width_mm", ""), "cage_width_mm", rown)
        entries.append(
            VideoEntry(
                video_name=name,
                source_path=src,
                intruder_entry_s=float(entry_s),
                crop=crop,
                downsample_width=ds_width,
                cage_width_mm=cage,
                cohort=str(row.get("cohort", "")),
                lighting=str(row.get("lighting", "")),
                notes=str(row.get("notes", "")),
            )
        )
    if not entries:
        raise ManifestError(f"video manifest {path} has no rows")
    return entries


def check_sources_exist(entries: Iterable[VideoEntry]) -> list[str]:
    """Return the names of manifest rows whose source video is missing."""
    return [e.video_name for e in entries if not e.source_path.exists()]


@dataclass
class AnnotationRow:
    video_name: str
    behavior: str
    start_s: float | None
    end_s: float | None
    annotator: str = ""


_SAFE_CLEAN = re.compile(r"[^A-Za-z0-9_\-]+")


def safe_video_name(name: str) -> str:
    """Turn a raw file name such as '10.3.25.ELS589' into the manifest form '10-3-25_ELS589'."""
    stem = Path(str(name).strip()).stem if str(name).lower().endswith((".mp4", ".avi", ".mov", ".mkv")) else str(name).strip()
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)\.(.+)$", stem)
    if m:
        stem = f"{m.group(1)}-{m.group(2)}-{m.group(3)}_{m.group(4)}"
    return _SAFE_CLEAN.sub("_", stem)


def parse_time_s(value, col: str = "time", row: int = 0, required: bool = False) -> float | None:
    """Seconds from '95', '95.5', '1:35', '1:35.5' or '0:01:35'."""
    if _blank(value):
        if required:
            raise ManifestError(f"row {row}: column {col!r} is required")
        return None
    txt = str(value).strip()
    if ":" in txt:
        parts = txt.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError as exc:
            raise ManifestError(f"row {row}: column {col!r} must be seconds or m:ss, got {value!r}") from exc
        if len(nums) > 3:
            raise ManifestError(f"row {row}: column {col!r} has too many ':' parts: {value!r}")
        total = 0.0
        for n in nums:
            total = total * 60 + n
        return total
    return _to_float(txt, col, row, required)


def video_name_lookup(cfg: PipelineConfig | None) -> dict[str, str]:
    """Map every known spelling of a processed video to its video_name."""
    lookup: dict[str, str] = {}
    if cfg is None:
        return lookup
    import json

    proc = cfg.path("paths.processed_videos")
    if not proc.exists():
        return lookup
    for sidecar in proc.glob("*.json"):
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = str(meta.get("video_name", sidecar.stem))
        cands = {name, sidecar.stem, str(meta.get("original_name", "")), Path(str(meta.get("source_path", ""))).stem,
                 Path(str(meta.get("source_path", ""))).name}
        for c in cands:
            if c:
                lookup[c.lower()] = name
                lookup[safe_video_name(c).lower()] = name
    return lookup


def resolve_video_name(name: str, lookup: dict[str, str]) -> str:
    """Return the pipeline video_name for an annotator's spelling, else its safe form."""
    key = str(name).strip()
    for cand in (key, safe_video_name(key)):
        hit = lookup.get(cand.lower())
        if hit:
            return hit
    return safe_video_name(key)


def read_annotations(path: str | Path, cfg: PipelineConfig) -> list[AnnotationRow]:
    """Read the annotation CSV, mapping behaviour aliases and shifting raw times to clip times."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"annotation file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    renames = {"video": "video_name", "file": "video_name", "behaviour": "behavior", "start": "start_s", "end": "end_s",
               "start_time": "start_s", "end_time": "end_s", "stop_s": "end_s", "stop": "end_s"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    for col in ("video_name", "behavior", "start_s", "end_s"):
        if col not in df.columns:
            raise ManifestError(f"annotation file is missing column {col!r} (have {list(df.columns)})")
    aliases = {str(k).strip().lower(): v for k, v in (cfg.get("annotations.behavior_aliases") or {}).items()}
    classifiers = cfg.classifiers
    for clf in classifiers:
        aliases.setdefault(clf.lower(), clf)
        aliases.setdefault(clf.lower().replace("_", " "), clf)
    none_token = str(cfg.get("annotations.none_token", "NONE")).upper()
    default_ref = str(cfg.get("annotations.time_reference", "processed")).strip().lower()
    lookup = video_name_lookup(cfg)
    entry_by_name: dict[str, float] = {}
    if default_ref == "raw" or "time_reference" in df.columns:
        import json

        for sidecar in cfg.path("paths.processed_videos").glob("*.json"):
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            entry_by_name[str(meta.get("video_name", sidecar.stem))] = float(meta.get("intruder_entry_s", 0.0))

    rows: list[AnnotationRow] = []
    for i, row in df.iterrows():
        rown = int(i) + 2
        raw_name = str(row["video_name"]).strip()
        if not raw_name:
            continue
        name = resolve_video_name(raw_name, lookup)
        beh_raw = str(row["behavior"]).strip()
        if beh_raw.upper() == none_token:
            rows.append(AnnotationRow(name, none_token, None, None, str(row.get("annotator", ""))))
            continue
        beh = aliases.get(beh_raw.lower())
        if beh is None:
            raise ManifestError(
                f"row {rown}: unknown behavior {beh_raw!r}; known: {sorted(set(aliases))} or {none_token}"
            )
        start = parse_time_s(row["start_s"], "start_s", rown, required=True)
        end = parse_time_s(row["end_s"], "end_s", rown, required=True)
        ref = str(row.get("time_reference", "") or default_ref).strip().lower()
        if ref == "raw":
            if name not in entry_by_name:
                raise ManifestError(f"row {rown}: time_reference raw but no processed sidecar for {name!r} to get the intruder-entry time")
            start -= entry_by_name[name]
            end -= entry_by_name[name]
            # a bout that ends before the intruder entered lies outside the clip
            if end <= 0:
                continue
            start = max(start, 0.0)
        elif ref not in ("processed", "clip", ""):
            raise ManifestError(f"row {rown}: time_reference must be 'processed' or 'raw', got {ref!r}")
        if end <= start:
            raise ManifestError(f"row {rown}: end_s must be greater than start_s")
        rows.append(AnnotationRow(name, beh, float(start), float(end), str(row.get("annotator", ""))))
    return rows

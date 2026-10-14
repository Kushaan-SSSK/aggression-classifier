# Kushaan Sharma

"""Read and validate the two hand-filled CSVs: the video manifest and the annotations.

Video manifest columns (config/video_manifest.template.csv):
    video_name        unique id, used as the file stem downstream (letters, digits, _ and -)
    source_path       raw video file, absolute or relative to the project root
    intruder_entry_s  seconds into the raw video when the intruder is placed in the cage
    crop_x, crop_y, crop_w, crop_h   crop rectangle in raw-video pixels, blank for no crop
    downsample        blank or 0 keeps the resolution, otherwise the target width in px
    cage_width_mm     inside cage width spanned by crop_w, blank uses simba.cage_width_mm
    clip_length_s     optional, seconds kept after the entry; blank uses video.clip_length_s
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

from ..config import LOG, PipelineConfig

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
    clip_length_s: float | None = None


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
        clip_len = _to_float(row.get("clip_length_s", ""), "clip_length_s", rown)
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
                clip_length_s=clip_len if clip_len and clip_len > 0 else None,
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


def behavior_aliases(cfg: PipelineConfig) -> dict[str, str]:
    """Lower-cased annotator spellings -> classifier name (or the ignore token)."""
    aliases = {str(k).strip().lower(): v for k, v in (cfg.get("annotations.behavior_aliases") or {}).items()}
    for clf in cfg.classifiers:
        aliases.setdefault(clf.lower(), clf)
        aliases.setdefault(clf.lower().replace("_", " "), clf)
    return aliases


def entry_times(cfg: PipelineConfig) -> dict[str, float]:
    """video_name -> intruder entry second in the raw recording, from the preprocessing sidecars."""
    import json

    out: dict[str, float] = {}
    proc = cfg.path("paths.processed_videos")
    if not proc.exists():
        return out
    for sidecar in proc.glob("*.json"):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        out[str(meta.get("video_name", sidecar.stem))] = float(meta.get("intruder_entry_s", 0.0))
    return out


def _shift_to_clip(name: str, start: float, end: float, ref: str, entries: dict[str, float], rown: int) -> tuple[float, float] | None:
    """Raw-recording seconds -> clip seconds (0 = intruder entry); None when the span lies before the entry."""
    if ref == "raw":
        if name not in entries:
            LOG.warning("row %d: %s not preprocessed yet, no intruder-entry time to shift raw times by; skipped", rown, name)
            return None
        start -= entries[name]
        end -= entries[name]
        if end <= 0:
            return None
        start = max(start, 0.0)
    elif ref not in ("processed", "clip", ""):
        raise ManifestError(f"row {rown}: time_reference must be 'processed' or 'raw', got {ref!r}")
    if end <= start:
        raise ManifestError(f"row {rown}: end_s must be greater than start_s")
    return float(start), float(end)


def _read_label_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"annotation file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    renames = {"video": "video_name", "file": "video_name", "behaviour": "behavior", "behaviors": "behavior", "behaviours": "behavior",
               "start": "start_s", "end": "end_s", "start_time": "start_s", "end_time": "end_s", "stop_s": "end_s", "stop": "end_s"}
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns and v not in df.columns})
    for col in ("video_name", "behavior", "start_s", "end_s"):
        if col not in df.columns:
            raise ManifestError(f"{path.name} is missing column {col!r} (have {list(df.columns)})")
    return df


def read_annotations(path: str | Path, cfg: PipelineConfig) -> list[AnnotationRow]:
    """Read the annotation CSV, mapping behaviour aliases and shifting raw times to clip times.

    EXCLUDE rows (annotations.exclude_token) are returned as they are: build_targets masks those frames.
    REMARK rows, rows whose alias is the ignore token and rows whose notes match
    annotations.ignore_if_notes_match for their behaviour contribute nothing; rows whose notes match
    annotations.exclude_if_notes_match become EXCLUDE rows; an unknown label is collected and reported
    once for the whole file.
    """
    df = _read_label_csv(path)
    aliases = behavior_aliases(cfg)
    none_token = str(cfg.get("annotations.none_token", "NONE")).upper()
    ignore_token = str(cfg.get("annotations.ignore_token", "IGNORE")).upper()
    exclude_token = str(cfg.get("annotations.exclude_token", "EXCLUDE")).upper()
    default_ref = str(cfg.get("annotations.time_reference", "processed")).strip().lower()
    lookup = video_name_lookup(cfg)
    entries = entry_times(cfg) if default_ref == "raw" or "time_reference" in df.columns else {}
    # e.g. {Attack: "low( to mid)? confidence"}: rows whose notes hedge this much are kept in the CSV but not trained
    ignore_notes = {k: re.compile(str(v), re.IGNORECASE) for k, v in (cfg.get("annotations.ignore_if_notes_match") or {}).items()}
    # ... and rows whose notes hedge like this are masked instead (EXCLUDE: neither positive nor negative)
    exclude_notes = {k: re.compile(str(v), re.IGNORECASE) for k, v in (cfg.get("annotations.exclude_if_notes_match") or {}).items()}

    rows: list[AnnotationRow] = []
    unknown: dict[str, list[int]] = {}
    for i, row in df.iterrows():
        rown = int(i) + 2
        raw_name = str(row["video_name"]).strip()
        if not raw_name:
            continue
        name = resolve_video_name(raw_name, lookup)
        beh_raw = str(row["behavior"]).strip()
        if beh_raw.upper() in ("REMARK", ignore_token):
            continue
        if beh_raw.upper() == none_token:
            rows.append(AnnotationRow(name, none_token, None, None, str(row.get("annotator", ""))))
            continue
        if beh_raw.upper() == exclude_token:
            beh = exclude_token
        else:
            beh = aliases.get(beh_raw.lower())
        if beh is None:
            unknown.setdefault(beh_raw, []).append(rown)
            continue
        if str(beh).upper() == ignore_token:
            continue
        hedge = ignore_notes.get(beh)
        if hedge is not None and hedge.search(str(row.get("notes", "") or "")):
            continue
        hedge = exclude_notes.get(beh)
        if hedge is not None and hedge.search(str(row.get("notes", "") or "")):
            beh = exclude_token
        start = parse_time_s(row["start_s"], "start_s", rown, required=True)
        end = parse_time_s(row["end_s"], "end_s", rown, required=True)
        ref = str(row.get("time_reference", "") or default_ref).strip().lower()
        span = _shift_to_clip(name, start, end, ref, entries, rown)
        if span is None:
            continue
        rows.append(AnnotationRow(name, beh, span[0], span[1], str(row.get("annotator", ""))))
    if unknown:
        listing = "; ".join(f"{k!r} (rows {', '.join(map(str, v))})" for k, v in unknown.items())
        raise ManifestError(f"{len(unknown)} unknown behavior label(s), add them to annotations.behavior_aliases: {listing}")
    return rows


@dataclass
class ScoredSpan:
    video_name: str
    classifiers: list[str]
    start_s: float
    end_s: float


def read_scored_spans(cfg: PipelineConfig, path: str | Path | None = None) -> list[ScoredSpan]:
    """Stretches of each video the annotator actually scored for a behaviour group (clip seconds).

    Columns: video_name, behaviors, start_s, end_s[, time_reference, notes]. `behaviors` is a group
    name from annotations.scored_span_groups or a comma-separated list of classifiers. A missing
    file means no partial scoring anywhere.
    """
    path = Path(path) if path else cfg.path("paths.scored_spans", "data/annotations/scored_spans.csv")
    if not path.exists():
        return []
    df = _read_label_csv(path)
    groups = {str(k).strip().lower(): list(v) for k, v in (cfg.get("annotations.scored_span_groups") or {}).items()}
    default_ref = str(cfg.get("annotations.time_reference", "processed")).strip().lower()
    lookup = video_name_lookup(cfg)
    entries = entry_times(cfg)
    spans: list[ScoredSpan] = []
    for i, row in df.iterrows():
        rown = int(i) + 2
        raw_name = str(row["video_name"]).strip()
        if not raw_name:
            continue
        name = resolve_video_name(raw_name, lookup)
        key = str(row["behavior"]).strip().lower()
        clfs = groups.get(key) or [c.strip() for c in str(row["behavior"]).split(",") if c.strip()]
        bad = [c for c in clfs if c not in cfg.classifiers]
        if bad:
            raise ManifestError(f"{path.name} row {rown}: unknown behaviors {bad}; use a group from annotations.scored_span_groups or classifier names")
        start = parse_time_s(row["start_s"], "start_s", rown, required=True)
        end = parse_time_s(row["end_s"], "end_s", rown, required=True)
        ref = str(row.get("time_reference", "") or default_ref).strip().lower()
        span = _shift_to_clip(name, start, end, ref, entries, rown)
        if span is None:
            continue
        spans.append(ScoredSpan(name, clfs, span[0], span[1]))
    return spans

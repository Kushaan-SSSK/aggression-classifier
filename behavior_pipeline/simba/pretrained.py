# Kushaan Sharma
"""List, download and run classifiers published on OSF by the Golden lab.
Run as the `pretrained` stage of the CLI (`--list`, `--download`, `--run`).
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from ..config import LOG, PipelineConfig

OSF_API = "https://api.osf.io/v2"
DEFAULT_NODE = "kwge8"


def osf_list(node: str = DEFAULT_NODE, path_url: str | None = None, prefix: str = "") -> list[dict]:
    """Recursively list files in an OSF node's osfstorage."""
    url = path_url or f"{OSF_API}/nodes/{node}/files/osfstorage/"
    items: list[dict] = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for entry in payload.get("data", []):
            attrs = entry.get("attributes", {})
            name = attrs.get("name", "")
            if attrs.get("kind") == "folder":
                sub = entry.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                if sub:
                    items.extend(osf_list(node, sub, prefix=f"{prefix}{name}/"))
            else:
                items.append({"name": name, "path": f"{prefix}{name}", "download": entry.get("links", {}).get("download"),
                              "size": attrs.get("size"), "modified": attrs.get("date_modified")})
        url = payload.get("links", {}).get("next")
        time.sleep(0.2)
    return items


def osf_download(items: list[dict], dest: Path, pattern: str | None = None, dry_run: bool = False) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for it in items:
        if pattern and pattern.lower() not in it["path"].lower():
            continue
        target = dest / it["path"]
        if target.exists() and it.get("size") and target.stat().st_size == it["size"]:
            out.append(target)
            continue
        LOG.info("%s %s (%s bytes)", "would download" if dry_run else "downloading", it["path"], it.get("size"))
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(it["download"], stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        out.append(target)
    return out


def download_pretrained(cfg: PipelineConfig, node: str = DEFAULT_NODE, pattern: str | None = ".sav", dry_run: bool = False) -> list[Path]:
    dest = cfg.path("paths.pretrained_models") / node
    items = osf_list(node)
    LOG.info("OSF node %s: %d file(s)", node, len(items))
    for it in items:
        LOG.info("  %s  (%s bytes)", it["path"], it.get("size"))
    return osf_download(items, dest, pattern=pattern, dry_run=dry_run)


def run_pretrained(cfg: PipelineConfig, model_paths: list[Path], threshold: float = 0.5, min_bout_ms: int = 100) -> Path:
    """Run downloaded .sav classifiers on this project's features with SimBA InferenceBatch."""
    from simba.model.inference_batch import InferenceBatch

    save_dir = cfg.simba_project_folder / "csv" / "machine_results_pretrained"
    save_dir.mkdir(parents=True, exist_ok=True)
    fps = cfg.fps
    model_dict = {}
    for p in model_paths:
        name = Path(p).stem
        model_dict[name] = {"model_path": str(Path(p).resolve()), "threshold": float(threshold),
                            "minimum_bout_length": int(round(min_bout_ms / 1000 * fps))}
    LOG.info("running %d pretrained classifier(s): %s", len(model_dict), list(model_dict))
    InferenceBatch(config_path=str(cfg.simba_config_path), save_dir=str(save_dir), model_dict=model_dict).run()
    return save_dir

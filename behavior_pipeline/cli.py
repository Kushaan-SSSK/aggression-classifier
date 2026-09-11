# Kushaan Sharma

"""Command-line entry point: python -m behavior_pipeline <stage> [options]

Stages run in .venv-simba except pose, which needs .venv-dlc.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_NAME, LOG, load_config, setup_logging


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help=f"path to pipeline.yaml, default ./{DEFAULT_CONFIG_NAME}")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behavior_pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("check", help="report which dependencies are available in this interpreter")
    _add_common(p)

    p = sub.add_parser("manifest-from-log", help="stage 0: write the video manifest from a SimBA batch_process_log.json")
    _add_common(p)
    p.add_argument("--batch-log", required=True, help="batch_process_log.json written by SimBA's batch tool")
    p.add_argument("--source-dir", required=True, help="folder holding the original recordings")
    p.add_argument("--out", default=None, help="manifest path, default paths.video_manifest")
    p.add_argument("--exclude", nargs="*", default=[], help="skip log entries whose name contains any of these tokens")
    p.add_argument("--overwrite", action="store_true", help="replace an existing manifest, keeping a .bak.csv copy")

    p = sub.add_parser("preprocess", help="stage 1: crop, trim, grayscale, 15 fps, CLAHE and frame counter per manifest row")
    _add_common(p)
    p.add_argument("--manifest", default=None, help="video manifest CSV, default paths.video_manifest")
    p.add_argument("--only", nargs="*", default=None, help="video_name(s) to process")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resident-clips", action="store_true", help="skip the short clips around intruder entry")

    p = sub.add_parser("adopt", help="stage 1b: ingest videos already processed by SimBA's batch tool")
    _add_common(p)
    p.add_argument("--video-dir", required=True, help="folder with the SimBA batch output videos")
    p.add_argument("--batch-log", default=None, help="batch_process_log.json, default next to the videos or one level up")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("pose", help="stage 2a: pose estimation with DeepLabCut SuperAnimal, run in .venv-dlc")
    _add_common(p)
    p.add_argument("--videos", nargs="*", default=None, help="video paths or globs, default all processed videos; with --bench, manifest video_name(s)")
    p.add_argument("--mode", choices=["blob", "detector"], default=None, help="blob uses background subtraction, detector uses Faster R-CNN; default from pose.mode")
    p.add_argument("--blob", action="store_true", help="alias for --mode blob")
    p.add_argument("--fast", action="store_true", help="detector mode: MobileNet-v3 detector instead of Faster R-CNN R50")
    p.add_argument("--box-score-thresh", type=float, default=None, help="detector mode: detector acceptance threshold, DLC default 0.6")
    p.add_argument("--boxes-only", action="store_true", help="blob mode: only compute boxes and a preview video to tune pose.blob")
    p.add_argument("--no-preview", action="store_true", help="blob mode: skip the labelled preview video")
    p.add_argument("--bench", action="store_true", help="benchmark preprocessing and detector variants on a short segment")
    p.add_argument("--segment-start", type=float, default=10.0, help="--bench: seconds after intruder entry the segment starts")
    p.add_argument("--segment-len", type=float, default=20.0, help="--bench: segment length in seconds")
    p.add_argument("--variants", nargs="*", default=None, help="--bench: subset of variants to run")
    p.add_argument("--status", action="store_true", help="list processed videos and whether a fresh pose file exists")

    p = sub.add_parser("resident", help="stage 2b: decide which tracked animal is the resident from the pre-entry frames")
    _add_common(p)
    p.add_argument("--videos", nargs="*", default=None, help="video_name(s), default all processed videos")
    p.add_argument("--preview", action="store_true", help="write a preview mp4 with the resident track")

    p = sub.add_parser("convert", help="stage 2c: pose H5/CSV to SimBA 16-body-part CSV")
    _add_common(p)
    p.add_argument("--videos", nargs="*", default=None, help="video_name(s), default all processed videos")
    p.add_argument("--swap", nargs="*", default=None, help="video_name(s) whose two animals must be flipped, overriding the resident stage")
    p.add_argument("--no-resident", action="store_true", help="ignore the resident stage's decisions")
    p.add_argument("--file", default=None, help="convert a single pose file, needs --name")
    p.add_argument("--name", default=None)
    p.add_argument("--keypoint-set", default=None, help="superanimal or lab_dlc, default from config")
    p.add_argument("--no-link", action="store_true", help="skip identity linking")

    p = sub.add_parser("project", help="stage 3: create the SimBA project, import pose, outliers, features")
    _add_common(p)
    p.add_argument("--overwrite", action="store_true", help="delete and recreate the SimBA project")
    p.add_argument("--step", choices=["create", "videos", "video_info", "import", "outliers", "features"], default=None)

    p = sub.add_parser("annotate", help="stage 4: annotations CSV to targets_inserted, or --make-todo to write the sheet to fill in")
    _add_common(p)
    p.add_argument("--annotations", default=None)
    p.add_argument("--make-todo", action="store_true", help="write data/annotations/annotations_todo.csv with one TODO row per processed video")

    p = sub.add_parser("train", help="stage 5: train random-forest classifiers")
    _add_common(p)
    p.add_argument("--clf", nargs="*", default=None)
    p.add_argument("--n-estimators", type=int, default=None)

    p = sub.add_parser("infer", help="stage 6: run classifiers to machine_results")
    _add_common(p)

    p = sub.add_parser("proximity", help="inter-animal proximity bouts")
    _add_common(p)

    p = sub.add_parser("aggression-proxy", help="attack-like bouts from pose alone, before any classifier exists")
    _add_common(p)
    p.add_argument("--videos", nargs="*", default=None, help="video_name(s), default all")

    p = sub.add_parser("unsupervised", help="stage 7: UMAP and HDBSCAN motif discovery")
    _add_common(p)
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--clf-slice", default=None, help="override unsupervised.clf_slice, e.g. Proximity, Attack or 'ALL CLASSIFIERS'")

    p = sub.add_parser("pretrained", help="list, download or run other labs' SimBA classifiers from OSF")
    _add_common(p)
    p.add_argument("--node", default="kwge8")
    p.add_argument("--list", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--pattern", default=".sav")
    p.add_argument("--run", nargs="*", default=None, help=".sav files to run on this project")

    p = sub.add_parser("synthetic", help="generate a synthetic project for a dry run")
    _add_common(p)
    p.add_argument("--out", default="synthetic_run", help="folder to create the synthetic project in")
    p.add_argument("--seconds", type=float, default=45.0)

    p = sub.add_parser("all", help="preprocess, resident, convert, project, annotate, train, infer, proximity, unsupervised")
    _add_common(p)
    p.add_argument("--skip", nargs="*", default=[], help="stages to skip")
    p.add_argument("--n-estimators", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.stage == "check":
        from env import check_env

        return check_env.main()
    cfg = load_config(args.config)
    LOG.info("config %s (root %s)", cfg.config_path, cfg.root)

    if args.stage == "manifest-from-log":
        from .video.adopt import load_batch_log, write_manifest_from_log

        out = Path(args.out) if args.out else cfg.path("paths.video_manifest")
        try:
            write_manifest_from_log(load_batch_log(Path(args.batch_log)), cfg, out, source_dir=Path(args.source_dir),
                                    exclude=tuple(args.exclude), overwrite=args.overwrite)
        except FileExistsError as exc:
            LOG.error("%s", exc)
            return 2
        print(out.read_text(encoding="utf-8"))
        return 0

    if args.stage == "preprocess":
        from .video.manifest import check_sources_exist, read_video_manifest
        from .video.preprocess import preprocess_all

        entries = read_video_manifest(args.manifest or cfg.path("paths.video_manifest"), cfg)
        if args.only:
            entries = [e for e in entries if e.video_name in set(args.only)]
        missing = check_sources_exist(entries)
        if missing:
            LOG.error("source videos missing for: %s", missing)
            return 2
        preprocess_all(entries, cfg, overwrite=args.overwrite, resident_clips=not args.no_resident_clips)
        return 0

    if args.stage == "adopt":
        from .video.adopt import adopt_all

        adopted = adopt_all(cfg, Path(args.video_dir), Path(args.batch_log) if args.batch_log else None, overwrite=args.overwrite)
        for name, meta in adopted.items():
            o = meta["output"]
            print(f"{name}\t{o['width']}x{o['height']}\t{o['fps']:.2f} fps\t{o['nb_frames']} frames\tentry {meta['intruder_entry_s']:.0f}s")
        return 0

    if args.stage == "pose":
        from .pose.superanimal import expand_video_args, pose_status, run_superanimal

        if args.status:
            for row in pose_status(cfg):
                print(f"{row['video']}\t{row['status']}\t{row['pose_file']}")
            return 0
        if args.bench:
            from .pose.bench import run_bench

            if not args.videos:
                LOG.error("--bench needs --videos <manifest video_name ...>")
                return 2
            run_bench(cfg, args.videos, segment_start_s=args.segment_start, segment_len_s=args.segment_len, variants=args.variants)
            return 0
        mode = "blob" if args.blob else (args.mode or str(cfg.get("pose.mode", "blob")))
        videos = expand_video_args(args.videos or [], cfg)
        if mode == "blob":
            from .pose.blob import run_blob_pose

            run_blob_pose(videos, cfg, preview=not args.no_preview, boxes_only=args.boxes_only)
        else:
            run_superanimal(videos, cfg, fast=args.fast, box_score_thresh=args.box_score_thresh)
        return 0

    if args.stage == "resident":
        from .pose.resident import run_resident

        for r in run_resident(cfg, args.videos, preview=args.preview):
            verdict = "undecided" if r["swap"] is None else ("Animal_2 is the resident -> swap" if r["swap"] else "Animal_1 is the resident")
            print(f"{r['video_name']}\t{verdict}\td1={r['d_animal1_px']} d2={r['d_animal2_px']} conf={r['confidence']}\t{r['reason']}")
        return 0

    if args.stage == "convert":
        from .pose.convert import convert_all, convert_pose_file

        if args.file:
            if not args.name:
                LOG.error("--file needs --name")
                return 2
            convert_pose_file(args.file, args.name, cfg, swap=bool(args.swap and args.name in args.swap),
                              keypoint_set=args.keypoint_set, link=not args.no_link)
        else:
            convert_all(cfg, args.videos, set(args.swap or []), use_resident=not args.no_resident)
        return 0

    if args.stage == "project":
        from .simba import project as sp

        if args.step is None:
            sp.build_project(cfg, overwrite=args.overwrite)
        else:
            {"create": lambda: sp.create_project(cfg, overwrite=args.overwrite), "videos": lambda: sp.import_videos(cfg),
             "video_info": lambda: sp.write_video_info(cfg), "import": lambda: sp.import_pose(cfg),
             "outliers": lambda: sp.outlier_correction(cfg), "features": lambda: sp.extract_features(cfg)}[args.step]()
        return 0

    if args.stage == "annotate":
        from .simba.annotations import build_targets, write_todo

        if args.make_todo:
            print(write_todo(cfg))
            return 0
        build_targets(cfg, args.annotations)
        return 0

    if args.stage == "train":
        from .simba.train import train_all

        train_all(cfg, args.clf, n_estimators=args.n_estimators)
        return 0

    if args.stage == "infer":
        from .simba.infer import infer_all

        infer_all(cfg)
        return 0

    if args.stage == "proximity":
        from .simba.proximity import compute_proximity

        print(compute_proximity(cfg).to_string(index=False))
        return 0

    if args.stage == "aggression-proxy":
        from .simba.aggression_proxy import compute_aggression_proxy

        print(compute_aggression_proxy(cfg, args.videos).to_string(index=False))
        return 0

    if args.stage == "unsupervised":
        from .simba.unsupervised import run_unsupervised

        if args.clf_slice:
            cfg.set("unsupervised.clf_slice", args.clf_slice)
        run_unsupervised(cfg, with_stats=not args.no_stats)
        return 0

    if args.stage == "pretrained":
        from .simba.pretrained import download_pretrained, osf_list, run_pretrained

        if args.list:
            for it in osf_list(args.node):
                print(f"{it['path']}\t{it.get('size')}\t{it.get('modified')}")
        if args.download:
            download_pretrained(cfg, node=args.node, pattern=args.pattern)
        if args.run:
            run_pretrained(cfg, [Path(p) for p in args.run])
        return 0

    if args.stage == "synthetic":
        from tests.synthetic import make_synthetic_project

        root = Path(args.out).resolve()
        make_synthetic_project(root, template_config=cfg.config_path, seconds=args.seconds)
        print(f"synthetic project created at {root}; run stages with --config {root / 'config' / 'pipeline.yaml'}")
        return 0

    if args.stage == "all":
        skip = set(args.skip)
        if "preprocess" not in skip:
            from .video.manifest import read_video_manifest
            from .video.preprocess import preprocess_all

            preprocess_all(read_video_manifest(cfg.path("paths.video_manifest"), cfg), cfg)
        if "resident" not in skip:
            from .pose.resident import run_resident

            run_resident(cfg)
        if "convert" not in skip:
            from .pose.convert import convert_all

            convert_all(cfg)
        if "project" not in skip:
            from .simba.project import build_project

            build_project(cfg)
        have_annotations = cfg.path("paths.annotations").exists()
        if not have_annotations:
            LOG.warning("%s not found: skipping annotate/train/infer", cfg.path("paths.annotations"))
        if have_annotations and "annotate" not in skip:
            from .simba.annotations import build_targets

            build_targets(cfg)
        if have_annotations and "train" not in skip:
            from .simba.train import train_all

            train_all(cfg, n_estimators=args.n_estimators)
        if have_annotations and "infer" not in skip:
            from .simba.infer import infer_all

            infer_all(cfg)
        if "proximity" not in skip:
            from .simba.proximity import compute_proximity

            compute_proximity(cfg)
        if "aggression-proxy" not in skip:
            from .simba.aggression_proxy import compute_aggression_proxy

            compute_aggression_proxy(cfg)
        if "unsupervised" not in skip:
            from .simba.unsupervised import run_unsupervised

            if not have_annotations:
                cfg.set("unsupervised.clf_slice", "Proximity")
            run_unsupervised(cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

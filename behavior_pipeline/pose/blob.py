# Kushaan Sharma

"""Pose from background-subtraction boxes and the SuperAnimal top-down head.

Run in .venv-dlc:  python -m behavior_pipeline pose --mode blob [--videos ...]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import LOG, PipelineConfig

DEFAULTS = {
    "bg_samples": 60,
    "bg_percentile": 90,
    "diff_threshold": 35,
    "dark_threshold": 60,  # None disables
    "blur": 7,
    "open_k": 9,
    "close_k": 9,
    "min_area_frac": 0.004,
    "min_area_rel": 0.15,  # of typical mouse area
    "carry_frames": 15,
    "max_area_frac": 0.15,
    "min_fill": 0.3,
    "max_box_frac": 0.5,
    "merge_area_factor": 1.3,
    "merge_len_factor": 1.8,
    "box_pad_frac": 0.20,
    "bg_window_s": 60,  # 0 = whole video
    "ignore_counter": True,
}

Box = tuple[int, int, int, int]


def _read_gray(cap) -> np.ndarray | None:
    import cv2

    ok, fr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr


def build_background(video: Path, n: int, percentile: float, frame_range: tuple[int, int] | None = None) -> np.ndarray:
    """Per-pixel percentile over n frames sampled evenly from frame_range (default: whole video)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = (0, max(N - 1, 0)) if frame_range is None else (max(frame_range[0], 0), min(frame_range[1], max(N - 1, 0)))
    frames = []
    for i in np.linspace(lo, hi, min(n, max(hi - lo + 1, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        g = _read_gray(cap)
        if g is not None:
            frames.append(g)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {video}")
    return np.percentile(np.stack(frames), percentile, axis=0).astype(np.uint8)


def build_backgrounds(video: Path, params: dict[str, Any]) -> tuple[list[np.ndarray], int]:
    """One background per window of bg_window_s seconds; returns (backgrounds, frames per window)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    win = int(float(params.get("bg_window_s", 0) or 0) * fps)
    if win <= 0 or win >= N:
        return [build_background(video, int(params["bg_samples"]), float(params["bg_percentile"]))], max(N, 1)
    bgs = []
    for k in range(int(np.ceil(N / win))):
        # each window is sampled from half a window before to half a window after
        rng = (k * win - win // 2, (k + 1) * win + win // 2)
        bgs.append(build_background(video, int(params["bg_samples"]), float(params["bg_percentile"]), frame_range=rng))
    return bgs, win


def counter_rect(frame_shape: tuple[int, ...], cfg: PipelineConfig | None) -> Box | None:
    """(x0, y0, x1, y1) of the frame-counter overlay drawn by the preprocess stage, or None."""
    if cfg is None:
        return None
    fc = cfg.get("video.frame_counter") or {}
    if not fc.get("enabled", True):
        return None
    import cv2

    font_scale = float(fc.get("font_scale", 0.7))
    position = str(fc.get("position", "top_left"))
    (tw, th), base = cv2.getTextSize("Frame: 000000", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)
    h, w = frame_shape[:2]
    margin, pad = 8, 6
    x0 = margin if "left" in position else max(w - tw - margin, 0)
    y_base = th + margin if "top" in position else h - margin
    return (max(x0 - pad, 0), max(y_base - th - pad, 0), min(x0 + tw + pad, w), min(y_base + base + pad, h))


def _split_blob(mask_component: np.ndarray) -> list[Box]:
    """Split one connected component into two boxes with 2-means on pixel coordinates."""
    ys, xs = np.nonzero(mask_component)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) < 20:
        return []
    import cv2

    _, labels, _ = cv2.kmeans(pts, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    boxes = []
    for k in (0, 1):
        p = pts[labels.ravel() == k]
        if len(p) < 10:
            continue
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        boxes.append((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)))
    return boxes


def detect_boxes(gray: np.ndarray, bg: np.ndarray, p: dict[str, Any], typical_area: float | None,
                 ignore_rect: Box | None = None) -> tuple[list[Box], float | None, int]:
    """Up to two boxes (x, y, w, h), the largest kept blob area, and the number of oversize blobs dropped."""
    import cv2

    diff = cv2.subtract(bg, gray)
    if ignore_rect is not None:
        x0, y0, x1, y1 = ignore_rect
        diff[y0:y1, x0:x1] = 0
    if p["blur"] > 1:
        diff = cv2.GaussianBlur(diff, (p["blur"], p["blur"]), 0)
    mask = diff > p["diff_threshold"]
    if p.get("dark_threshold") is not None:
        mask &= cv2.GaussianBlur(gray, (p["blur"], p["blur"]), 0) < int(p["dark_threshold"]) if p["blur"] > 1 else gray < int(p["dark_threshold"])
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((p["open_k"], p["open_k"]), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((p["close_k"], p["close_k"]), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = gray.shape
    min_area = p["min_area_frac"] * H * W
    max_area = float(p.get("max_area_frac", 1.0)) * H * W
    min_fill = float(p.get("min_fill", 0.0))
    max_box = float(p.get("max_box_frac", 1.0))
    comps_all = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    def _shape_ok(i: int, area: int) -> bool:
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        return area / max(bw * bh, 1) >= min_fill and bw <= max_box * W and bh <= max_box * H

    min_rel = float(p.get("min_area_rel", 0.0)) * typical_area if typical_area else 0.0
    comps = sorted([c for c in comps_all if min_rel <= c[0] <= max_area and _shape_ok(c[1], c[0])], reverse=True)
    n_oversize = len([c for c in comps_all if c[0] > max_area])
    if not comps:
        return [], None, n_oversize
    largest = comps[0][0]
    boxes: list[Box] = []
    if len(comps) == 1 and typical_area:
        # a single blob much bigger or longer than one mouse is two touching mice
        i0 = comps[0][1]
        longest = max(int(stats[i0, cv2.CC_STAT_WIDTH]), int(stats[i0, cv2.CC_STAT_HEIGHT]))
        too_big = largest > p["merge_area_factor"] * typical_area
        too_long = longest > float(p.get("merge_len_factor", 1.8)) * float(np.sqrt(typical_area))
        if too_big or too_long:
            boxes = _split_blob(lab == i0)
    if not boxes:
        for area, i in comps[:2]:
            x, y, w, h = (int(v) for v in stats[i, :4])
            boxes.append((x, y, w, h))
    padded = []
    for x, y, w, h in boxes:
        px, py = int(w * p["box_pad_frac"]), int(h * p["box_pad_frac"])
        x0, y0 = max(x - px, 0), max(y - py, 0)
        x1, y1 = min(x + w + px, W - 1), min(y + h + py, H - 1)
        padded.append((x0, y0, max(x1 - x0, 2), max(y1 - y0, 2)))
    return padded, float(largest), n_oversize


def fill_missing_boxes(boxes: list[list[Box]], frame_shape: tuple[int, int]) -> tuple[list[list[Box]], int]:
    """Give every frame at least one box: the previous frame's boxes, else the full frame."""
    # DeepLabCut drops frames with an empty context, which would misalign the H5 with the video.
    H, W = frame_shape
    filled: list[list[Box]] = []
    last: list[Box] = [(0, 0, int(W), int(H))]
    n_filled = 0
    for b in boxes:
        if b:
            last = list(b)
            filled.append(list(b))
        else:
            filled.append(list(last))
            n_filled += 1
    return filled, n_filled


def boxes_for_video(video: Path, params: dict[str, Any], cfg: PipelineConfig | None = None,
                    backgrounds: tuple[list[np.ndarray], int] | None = None) -> tuple[list[list[Box]], dict[str, Any]]:
    """Blob boxes per frame plus count statistics; backgrounds may be passed in from another video."""
    import cv2

    bgs, win = backgrounds if backgrounds is not None else build_backgrounds(video, params)
    ignore = counter_rect(bgs[0].shape, cfg) if params.get("ignore_counter", True) else None
    cap = cv2.VideoCapture(str(video))
    all_boxes: list[list[Box]] = []
    areas: list[float] = []
    typical: float | None = None
    n_oversize = 0
    n_carried = 0
    n_split = 0
    carry_limit = int(params.get("carry_frames", 0) or 0)
    last_two: list[Box] = []
    since_two = 0
    t = 0
    while True:
        g = _read_gray(cap)
        if g is None:
            break
        bg = bgs[min(t // win, len(bgs) - 1)]
        boxes, largest, dropped = detect_boxes(g, bg, params, typical, ignore_rect=ignore)
        n_oversize += dropped
        if len(boxes) == 1 and carry_limit and last_two and since_two < carry_limit:
            other = _carry_box(boxes[0], last_two)
            if other is not None:
                boxes = boxes + [other]
                n_carried += 1
            elif params.get("split_merged", True):
                # one blob covering both previous boxes: huddled mice, give the head two overlapping halves
                boxes = _split_box_geometric(boxes[0], last_two)
                n_split += 1
        all_boxes.append(boxes)
        if len(boxes) >= 2:
            last_two, since_two = list(boxes[:2]), 0
        else:
            since_two += 1
        if largest is not None and len(boxes) >= 2 and dropped == 0:
            areas.append(largest)
            if len(areas) >= 30:
                typical = float(np.median(areas[-300:]))
        t += 1
    cap.release()
    counts = np.array([len(b) for b in all_boxes])
    stats = {"n_frames": int(len(all_boxes)), "frames_0_boxes": int((counts == 0).sum()), "frames_1_box": int((counts == 1).sum()),
             "frames_2_boxes": int((counts == 2).sum()), "frames_second_box_carried": int(n_carried), "frames_merged_pair_split": int(n_split),
             "frames_oversize_blob_dropped": int(n_oversize), "typical_mouse_area_px": typical, "n_backgrounds": len(bgs),
             "counter_rect_ignored": ignore, "params": params}
    return all_boxes, stats


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)


def _covered(inner: Box, outer: Box) -> float:
    """Fraction of inner that lies inside outer."""
    ix0, iy0, iw, ih = inner
    ox0, oy0, ow, oh = outer
    ix = max(0, min(ix0 + iw, ox0 + ow) - max(ix0, ox0))
    iy = max(0, min(iy0 + ih, oy0 + oh) - max(iy0, oy0))
    return ix * iy / max(iw * ih, 1)


def _split_box_geometric(box: Box, last_two: list[Box] | None = None) -> list[Box]:
    """Two overlapping halves (60 % of the long side each) along the axis joining the previous two boxes."""
    x, y, w, h = box
    vertical = h >= w
    if last_two and len(last_two) == 2:
        (ax, ay, aw, ah), (bx, by, bw, bh) = last_two
        dx, dy = abs((ax + aw / 2) - (bx + bw / 2)), abs((ay + ah / 2) - (by + bh / 2))
        vertical = dy >= dx
    if vertical:
        half = max(int(round(h * 0.6)), 2)
        return [(x, y, w, half), (x, y + h - half, w, half)]
    half = max(int(round(w * 0.6)), 2)
    return [(x, y, half, h), (x + w - half, y, half, h)]


def _carry_box(current: Box, last_two: list[Box]) -> Box | None:
    """The previous box that does not correspond to current, or None when current contains both."""
    cover = [_covered(b, current) for b in last_two]
    if min(cover) > 0.5:
        return None
    ious = [_iou(current, b) for b in last_two]
    j = int(np.argmax(ious))
    return last_two[1 - j]


def build_pose_config(cfg: PipelineConfig, device: str = "cpu"):
    """Top-down SuperAnimal config; the detector name is needed for the crop preprocessor even though no detector runs."""
    from deeplabcut.pose_estimation_pytorch.config import PoseConfig

    sa = cfg.get("pose.superanimal") or {}
    sa_name = sa.get("superanimal_name", "superanimal_topviewmouse")
    model_name = sa.get("model_name", "hrnet_w32")
    det_name = sa.get("detector_name", "fasterrcnn_resnet50_fpn_v2")
    max_ind = int(sa.get("max_individuals", 2))
    config = PoseConfig.build_for_superanimal_inference(sa_name, model_name=model_name, detector_name=det_name,
                                                        max_individuals=max_ind, device=device)
    if not is_top_down(config):
        raise RuntimeError(f"SuperAnimal config method is {config.to_dict().get('method')!r}, expected top-down; boxes would be ignored")
    return config, sa_name, model_name, max_ind


def is_top_down(config) -> bool:
    """True for a top-down DLC config, whichever spelling of the method this DLC version uses."""
    method = str(config.to_dict().get("method", "")).upper().replace("-", "_")
    return "TOP_DOWN" in method or "TOPDOWN" in method or method == "TD" or method.endswith(".TD")


def run_pose_on_boxes(video: Path, boxes: list[list[Box]], cfg: PipelineConfig, device: str = "cpu") -> tuple[list[dict[str, Any]], list[str], str]:
    """Run the SuperAnimal pose head inside the given per-frame boxes; returns (predictions, bodyparts, scorer)."""
    import deeplabcut.pose_estimation_pytorch as pep
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path

    sa = cfg.get("pose.superanimal") or {}
    config, sa_name, model_name, max_ind = build_pose_config(cfg, device)
    snapshot = get_super_animal_snapshot_path(dataset=sa_name, model_name=model_name)
    runner = pep.get_pose_inference_runner(config, snapshot, batch_size=int(sa.get("batch_size", 8)), device=device, max_individuals=max_ind)
    try:
        # the sequential path keeps contexts aligned with frames
        runner.inference_cfg.multithreading.enabled = False
    except Exception:
        pass
    vi = pep.VideoIterator(str(video))
    vi.set_context([{"bboxes": np.array(b, dtype=float).reshape(-1, 4)} for b in boxes])
    preds = pep.video_inference(vi, runner)
    if len(preds) != len(boxes):
        raise RuntimeError(f"{video.name}: pose head returned {len(preds)} frames for {len(boxes)} box contexts")
    bodyparts = list(config.metadata.bodyparts)
    scorer = f"blobbox_{sa_name}_{model_name}"
    return preds, bodyparts, scorer


def predictions_to_h5(preds: list[dict[str, Any]], bodyparts: list[str], scorer: str, n_individuals: int, out_h5: Path | None) -> pd.DataFrame:
    """DLC-style multi-index frame; individuals without a valid pose become NaN with likelihood 0."""
    individuals = [f"animal{i}" for i in range(n_individuals)]
    cols = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ["x", "y", "likelihood"]],
                                      names=["scorer", "individuals", "bodyparts", "coords"])
    n_bp = len(bodyparts)
    data = np.full((len(preds), n_individuals * n_bp * 3), np.nan)
    for t, fr in enumerate(preds):
        arr = np.asarray(fr.get("bodyparts", np.zeros((0, n_bp, 3))))
        if arr.ndim == 2:
            arr = arr[None]
        for a in range(min(n_individuals, arr.shape[0])):
            kp = arr[a]
            if kp.shape[0] != n_bp:
                continue
            valid = kp[:, 2] > 0
            if not valid.any():
                continue
            data[t, a * n_bp * 3:(a + 1) * n_bp * 3] = kp[:, :3].reshape(-1)
    df = pd.DataFrame(data, columns=cols)
    lik = [c for c in df.columns if c[3] == "likelihood"]
    df[lik] = df[lik].fillna(0.0)
    if out_h5 is not None:
        df.to_hdf(out_h5, key="df_with_missing", mode="w", format="table")
    return df


def write_preview(video: Path, df: pd.DataFrame | None, boxes: list[list[Box]], out_mp4: Path, max_frames: int | None = None) -> None:
    """Labelled mp4 with the boxes and, when df is given, the keypoints."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colors = [(0, 0, 255), (0, 255, 0)]
    inds, xs, ys, ls = [], {}, {}, {}
    if df is not None:
        inds = list(df.columns.get_level_values("individuals").unique())
        xs = {a: df.xs((a, "x"), level=("individuals", "coords"), axis=1).to_numpy() for a in inds}
        ys = {a: df.xs((a, "y"), level=("individuals", "coords"), axis=1).to_numpy() for a in inds}
        ls = {a: df.xs((a, "likelihood"), level=("individuals", "coords"), axis=1).to_numpy() for a in inds}
    n_total = len(df) if df is not None else len(boxes)
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and t >= max_frames) or t >= n_total:
            break
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes[t] if t < len(boxes) else []:
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 200, 0), 1)
        for ai, a in enumerate(inds):
            for k in range(xs[a].shape[1]):
                if ls[a][t, k] > 0.3 and not np.isnan(xs[a][t, k]):
                    cv2.circle(fr, (int(xs[a][t, k]), int(ys[a][t, k])), 2, colors[ai % 2], -1)
        wr.write(fr)
        t += 1
    cap.release()
    wr.release()


def blob_params(cfg: PipelineConfig) -> dict[str, Any]:
    return {**DEFAULTS, **(cfg.get("pose.blob") or {})}


def run_blob_pose(videos: list[Path], cfg: PipelineConfig, dest: Path | None = None, preview: bool = True,
                  device: str = "cpu", boxes_only: bool = False) -> list[Path]:
    """Blob boxes and SuperAnimal pose per video; boxes_only writes just the box preview and stats."""
    dest = dest or cfg.path("paths.pose_raw")
    dest.mkdir(parents=True, exist_ok=True)
    params = blob_params(cfg)
    outputs = []
    for video in videos:
        video = Path(video)
        t0 = time.time()
        raw_boxes, stats = boxes_for_video(video, params, cfg)
        LOG.info("%s: boxes in %.0fs - 2 boxes in %d/%d frames, 1 box %d, none %d, oversize dropped %d", video.stem,
                 time.time() - t0, stats["frames_2_boxes"], stats["n_frames"], stats["frames_1_box"], stats["frames_0_boxes"],
                 stats["frames_oversize_blob_dropped"])
        import cv2

        cap = cv2.VideoCapture(str(video))
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        boxes, n_filled = fill_missing_boxes(raw_boxes, shape)
        stats["frames_boxes_filled_from_previous"] = n_filled
        if boxes_only:
            out_json = dest / f"{video.stem}_boxes.json"
            out_json.write_text(json.dumps({**stats, "boxes_per_frame": [len(b) for b in raw_boxes]}, indent=1), encoding="utf-8")
            write_preview(video, None, boxes, dest / f"{video.stem}_boxes_preview.mp4")
            LOG.info("%s: boxes-only -> %s", video.stem, out_json.name)
            outputs.append(out_json)
            continue
        preds, bodyparts, scorer = run_pose_on_boxes(video, boxes, cfg, device=device)
        out_h5 = dest / f"{video.stem}_{scorer}.h5"
        df = predictions_to_h5(preds, bodyparts, scorer, int(cfg.get("pose.superanimal.max_individuals", 2)), out_h5)
        stats["elapsed_s"] = round(time.time() - t0, 1)
        stats["sec_per_frame"] = round(stats["elapsed_s"] / max(len(df), 1), 3)
        stats["pct_frames_animal_tracked"] = {a: round(float((df.xs((a, "likelihood"), level=("individuals", "coords"), axis=1).max(axis=1) > 0.3).mean() * 100), 1)
                                              for a in df.columns.get_level_values("individuals").unique()}
        (dest / f"{video.stem}_{scorer}.blobstats.json").write_text(json.dumps({**stats, "boxes_per_frame": [len(b) for b in raw_boxes]}, indent=1), encoding="utf-8")
        if preview:
            write_preview(video, df, boxes, dest / f"{video.stem}_{scorer}_preview.mp4")
        LOG.info("%s: pose done in %.0fs -> %s (tracked %s)", video.stem, stats["elapsed_s"], out_h5.name, stats["pct_frames_animal_tracked"])
        outputs.append(out_h5)
    return outputs

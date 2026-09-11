# Pose-estimation and behaviour-analysis tool survey (September 2026)

Scope: open-source tools for two-mouse, top-down, resident-intruder recordings feeding SimBA.
Versions were checked on PyPI / GitHub on 2026-09-08.

## Recommendation

Use **DeepLabCut 3.0.1** in two steps:

1. **Zero-shot bootstrap** with the Model Zoo `superanimal_topviewmouse` model
   (`deeplabcut.video_inference_superanimal`, `max_individuals=2`). No labelling; 27 keypoints per
   mouse that include every point SimBA's 16-body-part scheme needs. Use it to check arena, lighting
   and preprocessing, and to get a first end-to-end result.
2. **Fine-tune** a multi-animal top-down model (`top_down_hrnet_w32` + Faster R-CNN detector, or
   SuperAnimal transfer learning via `build_weight_init` / `train_network(superanimal_transfer_learning=True)`)
   labelled directly in the 16-body-part scheme, across the lab's lighting conditions. This unlocks
   the Golden lab's published resident-intruder classifiers and SimBA's richest feature set.

Everything downstream (`convert` stage onwards) is identical for both.

## Tools considered

### DeepLabCut 3.0.1 (chosen)
- PyPI 2026-07-27; PyTorch backend (TensorFlow optional, to be dropped by 3.2). Python 3.10-3.12.
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (or a CUDA index) then `pip install "deeplabcut[modelzoo]"`.
- Multi-animal: bottom-up (DLCRNet, DEKR) and top-down (`top_down_<backbone>` + detector) architectures; trackers write `_el.h5` / `_sk.h5` / `_bx.h5`, which SimBA's `MADLCImporterH5` reads.
- SuperAnimal-TopViewMouse: 27 keypoints, ~5k training images (mostly C57BL/6, some CD1/white). Caveats: identity is per frame only; performance drops on white or mixed-coat pairs and unusual arenas; `video_adapt=True` improves jitter but fine-tunes for 1000 iterations (GPU only in practice).
- SimBA support: `import_dlc_csv_data`, `MADLCImporterH5`, `SuperAnimalTopViewImporter`.
- Windows 11: native; CUDA via PyTorch wheels; CPU works (~10x slower).

### SLEAP 1.6.5 (fallback)
- PyPI 2026-08-11; PyTorch (`sleap-nn`), Python 3.12/3.13, install via `uv tool install --python 3.13 "sleap[nn]" --torch-backend cpu|cu128`.
- Mature multi-animal top-down/bottom-up pipelines with identity models and a proofreading GUI.
- Output `.slp`, `sleap export -o analysis.h5` / `.csv`; SimBA imports all three.
- Equivalent capability to DLC 3 for this assay; would discard the lab's DLC labels and experience.

### SqueakPose Studio (docx link: github.com/dlhagger/SqueakPoseStudio)
- Real tool, not a misnaming: PyQt6 desktop app from Haggerty, Darden and Lovinger (NIAAA), bioRxiv 2026.01.24.700912 / eLife reviewed preprint 111308. Ultralytics YOLO26-pose (paper used YOLOv11s-pose), SAM-assisted masks, depth layer, UMAP/HDBSCAN utility, companion 3D-printed arena and Jetson acquisition box.
- Install: `git clone`, `uv sync`, `uv run python squeakpose_studio.py`; Python >= 3.12; CUDA/MPS/CPU.
- Multi-animal via per-class boxes and tracker IDs, but validated on single-animal open-field/operant data only.
- Output: YOLO labels and a per-frame CSV. No DLC/SLEAP export; SimBA can ingest through `SimBAYoloImporter` after column mapping.
- Verdict: promising for speed, immature for two-mouse resident-intruder work; academic licence.

### Others (one line each)
| Tool | Notes | Verdict |
|---|---|---|
| Lightning Pose 2.4.1 | Semi-supervised, single-animal, Linux/WSL + NVIDIA only | unsuitable |
| MMPose / RTMPose | Animal models (AP-10K etc.), no mouse model; DLC 3 already wraps RTMPose | unnecessary |
| Ultralytics YOLO11/26-pose | Custom labelling required; ByteTrack IDs; SimBA YOLO importer exists | DIY alternative |
| AlphaTracker | AlphaPose-based multi-mouse tracker, stale dependencies, no SimBA importer | skip |
| MARS v1.8 | End-to-end RI pipeline but hard-coded black resident / white intruder, TF1 era, last update 2021; SimBA has a 7-keypoint MARS importer | skip |
| idtracker.ai 6.0.14 | Identity-preserving centroid trajectories, no keypoints | possible identity oracle only |
| TRex 2.0 | Multi-animal tracking with visual ID, can run YOLO keypoints, no SimBA importer | skip |
| B-SOiD | UMAP+HDBSCAN motifs, single-animal, unmaintained since ~2021 | superseded by SimBA unsupervised |
| Keypoint-MoSeq 0.6.8 | Syllables from DLC/SLEAP/SuperAnimal keypoints, per-animal only, Windows GPU via WSL2 | secondary motif tool |
| VAME (EthoML fork) | VAE motifs, single-animal egocentric | skip for now |
| DeepEthogram 0.1.4 | Pixel-based supervised classification, not animal-resolved, NVIDIA GPU required | skip |

## SimBA (simba-uw-tf-dev 5.5.6, PyPI 2026-09-03)
- `pip install simba-uw-tf-dev`; Python 3.6 or 3.10 (3.10 used here); needs ffmpeg on PATH; requires `setuptools<81` (imports `pkg_resources`); a `dash` test plugin must be disabled for pytest (`-p no:dash`).
- Pose imports: DLC CSV, maDLC H5, SLEAP (slp/h5/csv), SuperAnimal-TopView, YOLO CSV, MARS, DANNCE, APT TRK, FaceMap, blob tracking.
- Built-in pose configs include "2 animals; 16 body-parts" (index 6).
- Headless API used by this pipeline: `simba.utils.config_creator.ProjectConfigCreator`, `simba.pose_importers.dlc_importer_csv.import_dlc_csv_data`, `simba.utils.cli.cli_tools.{set_outlier_correction_criteria_cli, feature_extraction_runner}`, `simba.outlier_tools.*`, `simba.model.train_rf.TrainRandomForestClassifier`, `simba.model.inference_batch.InferenceBatch`, `simba.data_processors.agg_clf_calculator.AggregateClfCalculator`, `simba.unsupervised.{dataset_creator, umap_embedder, hdbscan_clusterer, cluster_frequentist_calculator}`.
- GPU (cuML) acceleration is Linux/WSL only; everything used here is CPU.
- Published resident-intruder classifiers (Golden lab; Attack, Defensive, anogenital sniffing, lateral threat, mounting, pursuit, tail rattle; 16-bp, 2 animals): https://osf.io/kwge8/ (models), https://osf.io/tmu6y/ (project), https://osf.io/sr3ck/ (annotations), https://osf.io/5t4y9/ (DLC weights). They transfer only if the 16-bp scheme and comparable geometry/fps/px-per-mm are used; treat as pseudo-labelling seeds. Listing kwge8 through the OSF API (2026-09-08) shows the rat resident-intruder and CRIM13 `.sav` models are hosted there, but the mouse resident-intruder folder only says "SEE GOOGLE DRIVE FOLDER" (plus an example project zip); the mouse models must be fetched from the Golden lab Google Drive linked in the SimBA README.

## Windows 11 / GPU summary

| Tool | Native Win11 | GPU on Windows | CPU-only |
|---|---|---|---|
| DeepLabCut 3.0.1 | yes | CUDA via PyTorch wheels | yes (slow) |
| SLEAP 1.6.5 | yes (uv) | CUDA | yes |
| SimBA 5.5.6 | yes (primary dev OS) | NVENC video ops only; cuML needs WSL | yes (default) |
| SqueakPose Studio | yes | CUDA | yes |
| Lightning Pose | no (WSL) | WSL | no |
| Keypoint-MoSeq | CPU only | WSL2 experimental | yes |

## Sources
DeepLabCut docs (installation, ModelZoo, PyTorch user guide) and `deeplabcut/modelzoo/video_inference.py`; HuggingFace `mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse`; PyPI pages for deeplabcut, sleap, simba-uw-tf-dev, lightning-pose, idtrackerai; docs.sleap.ai; github.com/dlhagger/SqueakPoseStudio and its eLife/bioRxiv preprint; github.com/sgoldenlab/simba (README, docs/Scenario1.md, Scenario2.md, pseudoLabel.md, Multi_animal_pose.md, installation_new.md, notebooks) and simba-uw-tf-dev.readthedocs.io; OSF tmu6y / kwge8; MMPose animal model zoo; Ultralytics docs; AlphaTracker, MARS, TRex, B-SOiD, VAME, DeepEthogram, Keypoint-MoSeq repositories/docs.

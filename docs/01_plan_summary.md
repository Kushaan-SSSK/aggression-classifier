# Plan summary and design decisions

Source: `docs/Plan for Behavior Scoring.docx` ("Automated Behavioral Classification and Detection of
Novel Behavioral Patterns", last edited by Nicholas Russell, 2026-09-08). This page restates every
requirement in that document and records the decisions taken to implement them.

## What the docx asks for

Three major tasks and one minor task:

1. **Pose estimation** (tools named: DeepLabCut, SqueakPose Studio, SLEAP). Identify head/nose,
   flanks and rear of each mouse. Output must be SimBA-compatible (CSV, H5). Train under different
   lighting conditions. The lab currently uses DeepLabCut; alternatives are acceptable depending on
   their output. The embedded figure shows SimBA's "2 animals; 16 body-parts" scheme: per mouse
   Ear_left, Ear_right, Nose, Center, Lat_left, Lat_right, Tail_base, Tail_end.
2. **Video preprocessing** (Clipchamp, SimBA batch processing). Reduce data volume before pose
   estimation:
   - crop to the top-down view of the cage
   - trim to 5 minutes after the second mouse (intruder) is placed in the cage; longer videos are
     cut to the first 5 minutes, shorter ones (interventions) still start at intruder entry
   - downsample only if resolution/quality is poor
   - grayscale
   - 15 fps
   - superimpose a frame counter
   - CLAHE for contrast; some videos need saturation / colour balance fixed in Clipchamp first
3. **Supervised ML (SimBA)**: train a model to identify aggression and social investigation
   frame by frame; predict untrained frames; training set = clips with known aggression present or
   absent (Nick identifies and timestamps); optionally pseudo-label with other labs' models.
4. **Unsupervised ML (SimBA)**: find behavioural motifs/clusters around times of interest, i.e.
   experimenter-identified aggression bouts or times when the two mice are close together.

Links in the docx: DeepLabCut docs, `github.com/dlhagger/SqueakPoseStudio`, `sleap.ai`,
SimBA `docs/tutorial_process_videos.md`, SimBA readthedocs.

## Decisions

| Topic | Decision | Why |
|---|---|---|
| Pose engine | DeepLabCut 3.0.1 (PyTorch). Bootstrap with the zero-shot **SuperAnimal-TopViewMouse** model; switch to a fine-tuned maDLC model labelled in the 16-body-part scheme once labelled frames exist. | No labelled frames yet; lab already uses DLC; output is DLC H5; SimBA has native importers. See `02_pose_estimation_tool_survey.md`. |
| Body-part scheme | SimBA config index 6, "2 animals; 16 body-parts" | Matches the figure in the docx and the only scheme with published resident-intruder classifiers. |
| Classifiers | `Attack`, `Social_investigation` (edit in `config/pipeline.yaml`) | The two behaviours named in the docx. |
| Annotation format | Plain CSV (`video_name, behavior, start_s, end_s`) filled in Excel; converted directly into SimBA `targets_inserted` | Simplest thing Nick can produce from timestamps. BORIS files can still be imported with SimBA's own appender if preferred. |
| Preprocessing implementation | One ffmpeg pass (trim, crop, optional downsample, 15 fps, grayscale) + one OpenCV pass (CLAHE, frame counter) streamed into libx264 | Two encodes instead of six, and CLAHE is not an ffmpeg filter. SimBA's GUI batch tool remains available for one-off videos. |
| Identity | Frame-to-frame Hungarian re-linking on body centres, swap counts logged; the `resident` stage decides resident vs intruder from the frames before the intruder enters (only the resident is present); manual `--swap` remains as an override | Zero-shot models do not guarantee identity persistence for two same-coloured mice; the pre-entry frames are the one moment identity is unambiguous. |
| Pose on CPU (2026-09-11) | `pose --mode blob`: background-subtraction boxes + SuperAnimal HRNet top-down head on greyscale+CLAHE frames | The SuperAnimal detector finds nothing on greyscale and costs 1.4 s/frame; the pose head prefers greyscale (bench in `data/pose/bench/`). Blob boxes + head run at ~0.1 s/frame. |
| Environments | Python 3.10; `.venv-simba` (SimBA + pipeline) and `.venv-dlc` (DeepLabCut) | SimBA pins many exact versions; DLC needs torch. Both refuse Python 3.13. |
| Compute | CPU on this laptop (Intel Arc, no CUDA) | SuperAnimal on CPU is slow (tens of minutes per 5-min video). The same `pose` stage runs unchanged on a CUDA workstation or Google Colab (`notebooks/superanimal_colab.ipynb`). |
| Verification | Synthetic two-blob video + synthetic SuperAnimal H5 + derived annotations drive every stage in `pytest` | No real data exists yet. |

## Stage map (code in `behavior_pipeline/`)

| Docx task | Stage | Module | SimBA / DLC API used |
|---|---|---|---|
| (crop / entry times from the lab's SimBA batch log) | 0 `manifest-from-log` | `video/adopt.py` | reads `batch_process_log.json` |
| Preprocess videos | 1 `preprocess` | `video/preprocess.py` | ffmpeg, OpenCV (+ resident clips around intruder entry) |
| Pose estimation | 2a `pose` | `pose/blob.py` (`--mode blob`, default) / `pose/superanimal.py` (`--mode detector`) / `pose/bench.py` (`--bench`) | DLC `VideoIterator.set_context` + `video_inference` with the SuperAnimal HRNet head; `deeplabcut.video_inference_superanimal(max_individuals=2)` |
| Resident identity | 2b `resident` | `pose/resident.py` | blob track before intruder entry vs linked animal tracks |
| SimBA-compatible output | 2c `convert` | `pose/convert.py` | writes DLC-layout CSV in SimBA 16-bp column order |
| (project plumbing) | 3 `project` | `simba/project.py` | `ProjectConfigCreator`, `import_dlc_csv_data`, outlier correctors, `feature_extraction_runner` |
| Training set from timestamps | 4 `annotate` | `simba/annotations.py` | writes `csv/targets_inserted` |
| Supervised ML | 5 `train`, 6 `infer` | `simba/train.py`, `simba/infer.py` | `TrainRandomForestClassifier`, `InferenceBatch`, `AggregateClfCalculator` |
| Other labs' models | 5b `pretrained` | `simba/pretrained.py` | OSF download + `InferenceBatch(model_dict=...)` |
| Mice close together | `proximity` | `simba/proximity.py` | adds a `Proximity` bout column |
| Unsupervised ML | 7 `unsupervised` | `simba/unsupervised.py` | `DatasetCreator`, `UmapEmbedder`, `HDBSCANClusterer`, `ClusterFrequentistCalculator` |

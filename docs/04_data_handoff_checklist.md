# Data hand-off checklist: what is needed to continue

## Status 2026-09-11
The original recordings are now local (`videos/Training Videos/Training Videos/Preprocessed Videos/`,
1920x1080 @ 60 fps colour, side-by-side dual camera; the top-down cage is the right half) together
with four SimBA batch attempts and their `batch_process_log.json` files. The pipeline now runs from
those originals:

- `data/annotations/video_manifest.csv` was regenerated with `manifest-from-log` from the
  **Fourth attempt** log (the lab's latest crop rectangles); the clip start times in that log were
  checked frame by frame and are the intruder-entry moments (ELS580 5 s, ELS589 7 s, ELS693 13 s,
  ELS719 4 s, ELS663 7 s, ELS558 6 s).
- `data/processed_videos/` holds the five usable videos preprocessed by this pipeline (grey, CLAHE,
  15 fps, frame counter, 300 s or the full clip if shorter). `9-10-25_ELS558` is in the manifest but
  not processed: different camera geometry (1920x720 @ 30 fps).
- The earlier adopted (stretched-then-restored) videos were moved to `data/archive/processed_videos_adopted_2026-09-09/`
  and the 2026-09-09 pose output for ELS580 was deleted: it was produced by a bottom-up SuperAnimal
  config that ignored the boxes (see README "Known constraints").
- `data/resident_clips/` holds 6 s clips around each intruder entry; the `resident` stage uses the
  pre-entry frames (only the resident is present) to decide which tracked animal is the resident, so
  `convert --swap` is no longer needed by hand unless the stage reports "undecided".
- Pose runs on the CPU with `pose --mode blob` (~0.1 s/frame); `pose --bench` results are in
  `data/pose/bench/bench_results.csv`.

Results of the first full run (2026-09-11) are in `simba_project/resident_intruder/project_folder/logs/`
(`proximity_summary.csv`, `aggression_proxy_summary.csv`, `aggression_proxy/<video>_bouts.csv`) and
`unsupervised/`. Both mice carry a pose in 92-99.9 % of frames (`data/pose/simba_16bp/*.qc.json`). Resident
identity was decided automatically for all five (confidence 0.86-0.99, except **ELS663 at 0.66**: the resident
moved toward the intruder within the first second; please confirm with the lab which mouse is the resident in
8.3.26.ELS663). Note the raw animal order differs between pose runs, so `swap` in the resident JSON is per run.

Still open (nothing else is required to run the pipeline):
- [ ] **Behaviour timestamps.** `data/annotations/annotations_todo.csv` lists every processed video with
      a TODO row; replace the rows with Attack / Social_investigation bouts (or NONE) and save the result
      as `data/annotations/annotations.csv` (see `03_preprocessing_and_annotation_sop.md` section 3 and 5).
      Until then `train`/`infer` are skipped and the unsupervised stage slices on Proximity bouts.
- [ ] Cage width in mm along the crop's x-axis (config assumes 195 mm; only mm-scaled features and the
      40 mm proximity threshold depend on it).
- [ ] Names of 1-2 scored videos to hold out (`train.holdout_videos`).

## A. Raw videos (unblocks stage 1 preprocess, then 2 pose)
- [x] Original recordings, locally accessible.
- [x] `data/annotations/video_manifest.csv` (from the batch log; for new videos add rows by hand or run
      `manifest-from-log` on a new SimBA batch log with `--source-dir` pointing at the originals).
- [ ] A few videos from each lighting condition, so the pose model can be checked across them.

## B. Pose model choice (unblocks stage 2c convert and stage 3 project)
- [x] SuperAnimal zero-shot with blob boxes (`pose --mode blob`). Check `data/pose/raw/<video>_*_preview.mp4`
      and `data/pose/simba_16bp/<video>.qc.json` (`pct_frames_both_animals_have_center`, `n_jumps`).
- [ ] Optional later: a fine-tuned DLC model on labelled frames from these videos (the legacy lab model
      files are not available and were trained on the old red-light cage).

## C. Behaviour timestamps (unblocks stage 4 annotate, 5 train, 6 infer)
- [ ] `data/annotations/annotations.csv` (from `annotations_todo.csv`).
- [ ] Written operational definitions of the two behaviours (kept with the CSV).
- [ ] Names of 1-2 scored videos to hold out for validation.

## D. Optional
- [ ] Golden lab classifiers as pseudo-label seeds (`pretrained --list`; the mouse resident-intruder
      `.sav` files are on the lab's Google Drive, not OSF; download them by hand into `models/pretrained/`).
- [ ] A CUDA workstation / Colab is no longer needed for the blob route; it only matters for
      `pose --mode detector` with `video_adapt`.

## Run order
```
.venv-simba\Scripts\python -m behavior_pipeline manifest-from-log --batch-log <batch_process_log.json> --source-dir <originals> [--overwrite]
.venv-simba\Scripts\python -m behavior_pipeline preprocess [--only <names>]
.venv-dlc\Scripts\python   -m behavior_pipeline pose [--mode blob]        # or scripts\run_pose_bg.ps1
.venv-simba\Scripts\python -m behavior_pipeline resident
.venv-simba\Scripts\python -m behavior_pipeline convert [--swap <names>]
.venv-simba\Scripts\python -m behavior_pipeline project
.venv-simba\Scripts\python -m behavior_pipeline annotate      # needs annotations.csv
.venv-simba\Scripts\python -m behavior_pipeline train         # needs annotations.csv
.venv-simba\Scripts\python -m behavior_pipeline infer         # needs annotations.csv
.venv-simba\Scripts\python -m behavior_pipeline proximity
.venv-simba\Scripts\python -m behavior_pipeline unsupervised [--clf-slice Proximity]
.venv-simba\Scripts\python -m behavior_pipeline annotate --make-todo
```
`scripts\run_all.ps1` chains these and skips annotate/train/infer while `annotations.csv` is missing.

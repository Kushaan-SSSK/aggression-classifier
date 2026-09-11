# aggression-classifier

Scores mouse resident-intruder videos automatically: crops and trims the recordings, tracks both mice with
DeepLabCut, builds SimBA features, and trains classifiers for attack and social investigation. It also finds
close-contact bouts and behaviour motifs without labels.

Kushaan Sharma

## Setup (Windows)

```
powershell -ExecutionPolicy Bypass -File env\setup_env.ps1
```

This installs Python 3.10, ffmpeg and two environments: `.venv-simba` (SimBA) and `.venv-dlc` (DeepLabCut).

## Run

Put the recordings somewhere local, list them in `data/annotations/video_manifest.csv` (or make it from a
SimBA batch log with `manifest-from-log`), then:

```
.venv-simba\Scripts\python -m behavior_pipeline preprocess
.venv-dlc\Scripts\python   -m behavior_pipeline pose
.venv-simba\Scripts\python -m behavior_pipeline resident
.venv-simba\Scripts\python -m behavior_pipeline convert
.venv-simba\Scripts\python -m behavior_pipeline project
.venv-simba\Scripts\python -m behavior_pipeline proximity
.venv-simba\Scripts\python -m behavior_pipeline aggression-proxy
.venv-simba\Scripts\python -m behavior_pipeline unsupervised --clf-slice Proximity
```

`scripts\run_all.ps1` runs all of that in order.

Once scored timestamps exist in `data/annotations/annotations.csv` (see `docs/03_preprocessing_and_annotation_sop.md`):

```
.venv-simba\Scripts\python -m behavior_pipeline annotate
.venv-simba\Scripts\python -m behavior_pipeline train
.venv-simba\Scripts\python -m behavior_pipeline infer
```

Results land in `simba_project/resident_intruder/project_folder/logs/`.

## Layout

```
behavior_pipeline/   cli.py, config.py
  video/             manifest, preprocess, adopt
  pose/              blob, superanimal, bench, convert, resident
  simba/             project, annotations, train, infer, proximity, aggression_proxy, unsupervised, pretrained
config/pipeline.yaml every setting
docs/                plan, tool survey, annotation guide, handoff checklist, status update
tests/               unit tests and a synthetic end-to-end run (pytest tests)
```

# SOP: preparing videos and timestamps for the pipeline

Audience: whoever prepares the recordings and scores behaviour (the docx names Nick).
Nothing in this SOP needs Python; the outputs are two CSV files and a folder of videos.

## 1. Video files

1. Copy the raw recordings into `data/raw_videos/`. Any container ffmpeg reads is fine (mp4, avi,
   mov, mkv). Do not rename them after the manifest is written.
2. Give each recording a **video_name**: letters, digits, `_` and `-` only, no spaces, unique.
   Suggested pattern `cohortA_res01_int07` (cohort, resident id, intruder id).
3. If a video has a colour cast or very low contrast, fix saturation / warm-cold balance in
   Clipchamp first and save the corrected file; the pipeline's CLAHE step handles ordinary contrast.

### If you use SimBA's own batch-processing GUI instead
It works (crop, clip at intruder entry, 15 fps, grayscale, CLAHE, frame counter) with one trap: SimBA
crops first and *then* applies "downsample", so leave the downsample box **unticked** (or type the crop
width/height into it). With 1920x1080 left in the boxes the cropped cage is stretched back to full HD.
Keep the `batch_process_log.json` it writes next to the output videos; the pipeline's `adopt` stage reads it.

## 2. Video manifest (`data/annotations/video_manifest.csv`)

Start from `config/video_manifest.template.csv`. One row per video:

| column | what to enter |
|---|---|
| `video_name` | as above |
| `source_path` | path to the raw file, e.g. `data/raw_videos/cohortA_res01_int07.mp4` |
| `intruder_entry_s` | seconds from the start of the raw video to the moment the intruder is set down in the cage. Scrub in any player; note minutes:seconds and convert (2:35 -> 155). Precision of about half a second is enough. |
| `crop_x, crop_y, crop_w, crop_h` | crop rectangle in raw-video pixels covering the cage floor from the top-down view. Read the pixel coordinates of the top-left corner and the width/height from any image viewer on a screenshot (or leave all four blank to keep the full frame). One rectangle per camera setup is usually reused for a whole cohort. |
| `downsample` | leave blank. Enter a target width in pixels (e.g. 640) only for videos the docx says to downsample (poor quality). |
| `cage_width_mm` | inside width of the cage covered by `crop_w`, in mm (default 195 in config, an unconfirmed assumption for the current cage). This calibrates distances. |
| `cohort`, `lighting`, `notes` | free text; `lighting` (e.g. bright / dim / red) is used later to check the pose model across conditions |

The pipeline then produces `data/processed_videos/<video_name>.mp4`: cropped, starting at intruder
entry, at most 5 minutes long, grayscale, 15 fps, CLAHE, frame counter top-left, plus a `.json`
sidecar recording every parameter.

## 3. Behaviour timestamps (`data/annotations/annotations.csv`)

Start from `config/annotations.template.csv`. **All times are seconds in the processed video**,
i.e. 0 = intruder entry, and the frame counter burned into the processed video divided by 15
gives the time. Score from the processed video, not the raw one.

| column | what to enter |
|---|---|
| `video_name` | must match the manifest |
| `behavior` | `Attack` or `Social_investigation` (aliases accepted: attack, aggression, fight, sniff, investigation). Use `NONE` with blank times for a video that was scored completely and contained no events. |
| `start_s`, `end_s` | bout start and end in seconds (decimals allowed) |
| `annotator` | initials |
| `notes` | optional |

Rules that matter for training quality:
- Score whole videos, not only the interesting minutes; frames outside listed bouts are treated as
  "behaviour absent", so unscored stretches would teach the model that attacks are absent there.
- Apply one written operational definition per behaviour consistently across videos.
- Bouts of the two behaviours may overlap; each classifier is trained independently.
- 8 to 12 fully scored videos with a good number of bouts is a reasonable first training set;
  keep 1 or 2 scored videos out of training for validation (`train.holdout_videos` in config).

### If the crop rectangles and entry times already exist in a SimBA batch log
`python -m behavior_pipeline manifest-from-log --batch-log <batch_process_log.json> --source-dir <folder with the originals>`
writes the manifest from the log's crop rectangles and clip start times (clip start = intruder entry).

## 4. Pose sanity check (after the pose stage)

The pose stage writes a preview video next to the H5 output (`data/pose/raw/<video>_*_preview.mp4`:
boxes in yellow, Animal_1 keypoints red, Animal_2 green). Open it and confirm:
- both mice are detected for most of the video (`pct_frames_both_animals_have_center` in
  `data/pose/simba_16bp/<video_name>.qc.json`; `n_jumps` counts suspicious identity jumps);
- Animal_1 is the resident. The `resident` stage decides this from the frames before the intruder
  enters (`data/pose/resident/<video>.resident.json`, `swap_source` in the qc.json). If it reports
  "undecided", check the preview and re-run `convert --swap <video_name>` if Animal_1 is the intruder.

## 5. Annotation timing for the current videos

`python -m behavior_pipeline annotate --make-todo` writes `data/annotations/annotations_todo.csv`
with one `TODO` row per processed video and the clip length in the notes. To score:

1. Open `data/processed_videos/<video_name>.mp4` (this is the clip that starts at intruder entry).
2. For each bout note start and end as **seconds in this clip** = frame counter (top-left) / 15.
3. Replace the TODO row with one row per bout (`Attack` or `Social_investigation`); if a video was
   watched completely and has no bouts, replace it with a single `NONE` row (blank times).
4. Save the finished sheet as `data/annotations/annotations.csv`. The pipeline refuses files that
   still contain TODO rows, so nothing can be trained on a half-finished sheet.

The reader is tolerant of the usual spreadsheet conventions, so a sheet in the annotator's own
format can be used directly:
- video names may be the original file names (`10.3.25.ELS580`, with or without `.mp4`);
- column headings `Video / Behaviour / Start / End` are accepted;
- times may be seconds (`95`, `95.5`) or `m:ss` (`1:35`, `1:35.5`);
- behaviour names are matched case-insensitively against `annotations.behavior_aliases` in
  `config/pipeline.yaml` (attack, aggression, fight, biting, ... -> Attack; social investigation,
  sniffing, investigating, ... -> Social_investigation); add new spellings there;
- if the timestamps were taken in the **untrimmed** recording (from the start of the raw video
  rather than from intruder entry), set `annotations.time_reference: raw` in the config, or add a
  `time_reference` column with `raw` on those rows: the pipeline subtracts each video's
  intruder-entry time and drops bouts that end before the intruder entered.

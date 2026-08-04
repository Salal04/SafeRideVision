# Safe Ride Vision — Backend

The inference backend for Safe Ride Vision: a Flask server (running on Colab, exposed via ngrok) that takes an uploaded video, runs it through the full detection/tracking pipeline, and returns an annotated output video + a per-frame violation log. This is what the [frontend](./README.md)'s "Run Detection" page talks to.

## DEMO

## Files

| File | Role |
|---|---|
| `backend_server.py` | Flask app — HTTP layer only. Saves uploads, converts frontend "junctions" into pipeline "turn zones", calls the pipeline, serves results back. |
| `deepsortVideo.py` | The actual pipeline: loads all models, runs detection → tracking → mirror/indicator/orientation → blink → verdict per frame, writes the annotated video + CSV log. |
| `track.py` | The `Track` class (extends Deep SORT's base track) — per-bike state: trajectory history, indicator slots, turn detection, box color. |
| `deep_sort/` | The nwojke/deep_sort tracker package (Kalman filter + Hungarian assignment) — expected alongside these files, not included here. |
| `blink_model.py` | ResNet18 + LSTM blink classifier definition (`build_model`, `load_weights`) — expected alongside these files. |

## How a request flows

1. **`POST /process-video`** hits `backend_server.py`. It saves the video, reads `mode` (`direct`/`junction`), and — if `junction` — converts the frontend's drawn shapes (rectangle/circle/polygon, in native frame pixels) into the `TURN_ZONES` format `deepsortVideo.py` expects. In `direct` mode `TURN_ZONES = []`.
2. It calls `deepsortVideo.run_video(video_path, output_dir)`.
3. `run_video()` loads all models **once at import time** (not per-request) and then, frame by frame:
   - Detects motorbikes (YOLO), runs them through NMS, extracts OSNet ReID embeddings, and updates the Deep SORT tracker.
   - Detects mirrors, indicators, and orientation (side/front-back) — each with its own YOLO model — and matches each detection to the nearest live track by IoU/center-containment (`best_match_track`).
   - Computes RAFT optical flow between consecutive frames for extra per-bike motion stats.
   - For each live track: updates its trajectory, runs the angle-based turn detector (`Track.detect_turn`), **and** checks zone overlap (`zone_overlap_ratio` — a no-op when no zones are configured) — either one flagging a turn is enough (OR logic).
   - Matches indicator detections to two persistent left/right "slots" per bike (`Track.update_indicators`), carrying a slot forward by the bike's own movement on frames where it's missed, so brief occlusion doesn't reset it.
   - If a bike is turning, crops each active indicator slot, feeds its accumulated crop history into the ResNet+LSTM blink model, and gets a blinking/not-blinking verdict per slot.
   - Sets box color from the combined state: **green** = not turning, **red** = turning with no confirmed signal, **black** = turning with a confirmed signal.
   - Appends one CSV row per live track this frame, draws all boxes/trails/zones onto the frame, and writes it to the output video.
4. After the loop, the raw `mp4v` output is re-encoded to H.264 with `ffmpeg` (falls back to the raw file if `ffmpeg` isn't available) so it plays in a browser `<video>` tag instead of only desktop players.
5. `backend_server.py` parses the CSV log into JSON, records the request in a manifest, and returns `{ output_video_url, logs }`.

## Turn detection: two signals, OR'd together

- **Angle-based** (`Track.detect_turn`, always on): keeps a rolling buffer of the bike's trajectory, smooths it, and sums the absolute frame-to-frame heading change. Above threshold (and enough side/front-back orientation votes) → flagged as turning. Works with zero configuration.
- **Zone-based** (`TURN_ZONES`, only when the frontend sends `mode: "junction"`): a static mask is rasterized once per video from the user-drawn zones; if a bike's box overlaps a zone above `TURN_ZONE_OVERLAP_THRESHOLD` (30% by default), it's flagged as turning too.

Either signal alone is enough — they're not required to agree.

## Compliance rule

For every frame a bike is flagged as turning, its two indicator slots are checked by the blink model:

- Turning + confirmed blink on either slot → **compliant** (logged `TO`)
- Turning + no confirmed blink → **violation** (logged `TF`)
- Not turning → not evaluated (logged `N`)

## API

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check. |
| `/process-video` | POST | Runs the full pipeline on an uploaded video. See request/response shape below. |
| `/videos` | GET | Lists all previously processed videos (manifest), sorted by upload time. Query params: `order` (`asc`/`desc`), `limit`. |
| `/outputs/<filename>` | GET | Static file serving for processed output videos. |

**`POST /process-video`** — multipart/form-data:
- `video` — the file (required)
- `mode` — `"direct"` or `"junction"`
- if `mode == "junction"`: `fps`, `junction_count`, `junctions` (JSON array of `{shape, box, points, frame_number, timestamp_sec, frame_width, frame_height}`)

Response:
```json
{
  "output_video_url": "/outputs/20260804-101500_clip.mp4",
  "logs": [
    {"frame_idx": "128", "track_id": "42", "cx": "612.0", "cy": "340.5",
     "orientation_side": "4", "orientation_frontback": "1",
     "flow_dx": "1.204", "flow_dy": "-0.312", "flow_magnitude": "1.243", "flow_angle": "-14.5",
     "cum_angle_change": "42.10", "zone_overlap_ratio": "0.000",
     "is_turning_algo": "True", "human_verification": "not turn"}
  ],
  "mode": "direct",
  "junction_count": 0,
  "received_at": 1754308500.12
}
```
CSV log columns (one row per live track per frame): `frame_idx, track_id, cx, cy, orientation_side, orientation_frontback, flow_dx, flow_dy, flow_magnitude, flow_angle, cum_angle_change, zone_overlap_ratio, is_turning_algo, human_verification`. The last column is a placeholder for manual review, not filled in by the pipeline.

## Models used

| Model | Task | Notes |
|---|---|---|
| YOLO (`YOLO_MODEL`) | Motorbike detection | e.g. `yolov10m.pt` |
| Mirror detector | Locates side mirrors | Old combined mirror/indicator model, mirror class only |
| Indicator detector | Locates indicator lamp | YOLOv12m, single "indicator" class |
| Orientation detector | Side vs front-back pose | Custom-trained YOLO |
| OSNet (`torchreid`) | Re-ID embeddings for Deep SORT | Trained on a bike-specific dataset — square 256×256 input, **not** the pedestrian-standard 256×128 |
| RAFT-small (`torchvision`) | Optical flow between frames | Swapped in for FlowNet2 (avoids compiling a custom CUDA correlation layer on Colab) |
| ResNet18 + LSTM (`blink_model.py`) | Blink vs static-on classification | Reads each indicator slot's accumulated crop history |

All model weight paths are set as constants near the top of `deepsortVideo.py` (`YOLO_MODEL`, `MIRROR_DETECTOR`, `INDICATOR_DETECTOR`, `Orientation_detector_path`, `REID_WEIGHTS`, and the blink model's weights path) — update these to wherever your weight files actually live before running.

## Running it (Colab)

1. Mount Drive and unzip the project bundle into `/content/deepSort`.
2. `pip install ultralytics flask flask-cors pyngrok`
3. `git clone https://github.com/KaiyangZhou/deep-person-reid.git` (provides `torchreid`)
4. Make sure `deep_sort/` (with the `preprocessing.py` NMS fix) and `blink_model.py` are in place alongside `deepsortVideo.py`.
5. Set your own ngrok authtoken (`ngrok config add-authtoken <your-token>` — don't reuse anyone else's).
6. `%cd /content/deepSort` then `python backend_server.py`.
7. Copy the printed ngrok URL into the frontend's Upload page → Backend connection field.

GPU strongly recommended — the pipeline runs 5 models (YOLO ×4, RAFT, OSNet, blink LSTM) per frame.

## Known constraints / things to double check

- Model weight paths are hardcoded to specific Drive locations — broken paths fail loudly at import time (models load once, at startup).
- `MAX_COSINE_DISTANCE` (Deep SORT's ReID matching threshold) was tuned for the old pedestrian encoder; re-tune empirically now that OSNet's bike-specific embedding space is in use.
- No auth on any route — anyone with the ngrok URL can hit `/process-video`. Fine for a demo/dev tunnel, not for anything public-facing.
- No request queueing — concurrent uploads run through the same loaded models on whatever thread Flask hands them; fine for solo/demo use, would need a job queue under real concurrent load.

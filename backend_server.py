"""
Safe Ride Vision — inference backend (Colab + ngrok)
=====================================================

Matches the contract your frontend's api.js already expects:

    POST {ngrok_url}/process-video   (multipart/form-data)
      "video"           -> the video file           [always]
      "mode"            -> "direct" | "junction"    [always]
      "fps"             -> float                    [only if mode == "junction"]
      "junction_count"  -> int                       [only if mode == "junction"]
      "junctions"       -> JSON array (see below)     [only if mode == "junction"]

    Response JSON:
      {
        "output_video_url": "/outputs/<file>.mp4",   // resolved by api.js against the ngrok base
        "logs": [ {frame_idx, track_id, cx, cy, ...}, ... ]
      }

WHAT THIS FILE DOES
--------------------
1. Saves the uploaded video to disk.
2. Turns the frontend's `junctions` (rectangle/circle/polygon, in native
   frame pixels) into the TURN_ZONES format `deepsortVideo.py` already
   knows how to rasterize and check bikes against.
   - mode == "direct"            -> TURN_ZONES = []  (angle-detector only,
                                     exactly the "no polygon" pipeline path)
   - mode == "junction" + zones  -> TURN_ZONES = [...]  (zone-overlap ALSO
                                     counts as a turn, on top of the angle
                                     detector — same OR logic already in
                                     deepsortVideo.py, nothing to change there)
3. Runs `deepsortVideo.run_video()` on the saved file.
4. Moves the processed .mp4 into a public "output_public" folder and
   serves it back over a static route, so the frontend can just point a
   <video> tag at output_video_url.
5. Parses the pipeline's per-frame CSV log into JSON so it can ride along
   in the same response (api.js already reads `data.logs`).
6. Opens an ngrok tunnel and prints the URL to paste into the frontend.

SETUP (run once in a Colab cell before this file):
    !pip install flask flask-cors pyngrok -q

This file expects `deepsortVideo.py`, `track.py`, the `deep_sort/` package,
`blinker`, and all the model weight files it references to already be in
place exactly as `deepsortVideo.py` was written for — this backend just
calls into that pipeline, it doesn't change how detection/tracking works.
"""

import os
import json
import time
import shutil
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# The DeepSORT/turn/indicator pipeline uploaded earlier. Keep this file
# next to it (or on PYTHONPATH). Importing it loads YOLO / OSNet / the
# blink-detector / RAFT once, at process startup — NOT per-request — so
# requests after the first one aren't stuck waiting on model loading.
import deepsortVideo as pipeline

# ----------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------
BASE_DIR = Path("/content/deepSort")
UPLOAD_DIR = BASE_DIR / "uploads"              # incoming videos land here
OUTPUT_DIR = BASE_DIR / "output_public"        # <-- served statically, this is what the frontend hits
RUN_LOG_DIR = BASE_DIR / "run_logs"            # per-request CSV logs

for d in (UPLOAD_DIR, OUTPUT_DIR, RUN_LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Point the pipeline's own log directory at ours, so everything lives
# under one tree instead of the Drive path hardcoded at the top of
# deepsortVideo.py.
pipeline.LOG_DIR = str(RUN_LOG_DIR)

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = Flask(__name__, static_folder=None)
CORS(app)  # the ngrok URL is cross-origin from the frontend, so allow it

# ----------------------------------------------------------------------
# Manifest — one entry per processed video, in the order it was RECEIVED
# (i.e. upload time), so /videos can hand the frontend everything sorted
# the same way regardless of how long each video took to process.
# ----------------------------------------------------------------------
MANIFEST_PATH = BASE_DIR / "manifest.json"
_manifest_lock = threading.Lock()  # process_video() can run from multiple request threads


def _load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _append_manifest(entry):
    with _manifest_lock:
        entries = _load_manifest()
        entries.append(entry)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    return entry


# ----------------------------------------------------------------------
# Junction (frontend) -> Turn zone (pipeline) conversion
# ----------------------------------------------------------------------
def _junction_to_zone(j):
    """One entry of the frontend's `junctions` array -> one dict in the
    shape deepsortVideo.build_zone_mask() / draw_zones() expect.

    Frontend shapes: "rectangle" | "circle" | "polygon", box is always the
    bounding box in native frame pixels; points is only present (and only
    used) for polygons.
    """
    shape = j.get("shape")
    box = j.get("box") or {}

    if shape == "circle":
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))
        cx, cy = x + w / 2.0, y + h / 2.0
        radius = max(w, h) / 2.0
        if radius <= 0:
            return None
        return {"type": "circle", "center": (cx, cy), "radius": radius}

    if shape == "polygon":
        pts = j.get("points") or []
        points = [(float(p["x"]), float(p["y"])) for p in pts]
        if len(points) < 3:
            return None  # degenerate, not enough vertices to fill
        return {"type": "polygon", "points": points}

    # default: "rectangle"
    x = float(box.get("x", 0))
    y = float(box.get("y", 0))
    w = float(box.get("width", 0))
    h = float(box.get("height", 0))
    if w <= 0 or h <= 0:
        return None
    return {"type": "rectangle", "points": [(x, y), (x + w, y + h)]}


def _parse_turn_zones(form):
    """[] whenever mode != 'junction' or nothing usable was sent — that's
    the signal deepsortVideo.py's TURN_ZONES already treats as "run on
    angle-based turn detection alone" (build_zone_mask returns None,
    zone_overlap_ratio is always 0.0)."""
    if form.get("mode", "direct") != "junction":
        return []

    raw = form.get("junctions")
    if not raw:
        return []

    try:
        junctions = json.loads(raw)
    except (TypeError, ValueError):
        return []

    zones = []
    for j in junctions:
        try:
            zone = _junction_to_zone(j)
        except (TypeError, ValueError, KeyError):
            zone = None
        if zone is not None:
            zones.append(zone)
    return zones


def _parse_log_file(log_path):
    """The pipeline's CSV (header written once by run_video, one row per
    live track per frame) -> list[dict], so it can travel in the JSON
    response the same way api.js already expects (`data.logs`)."""
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if len(lines) < 2:
        return []

    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue  # skip a malformed row rather than 500 the whole response
        rows.append(dict(zip(header, parts)))
    return rows


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    """Public static access to processed videos. api.js resolves
    output_video_url against the same ngrok base, so a relative path
    like '/outputs/xyz.mp4' just works."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/videos", methods=["GET"])
def list_videos():
    """All processed videos, sorted by the time they were RECEIVED
    (upload time) — not by when processing finished, so a slow big
    upload doesn't jump ahead of / behind a quick small one that came in
    after it.

    Query params:
      order   "asc" (default, oldest received first) | "desc" (newest first)
      limit   optional int — only return the last N entries after sorting
    """
    order = request.args.get("order", "asc").lower()
    entries = sorted(_load_manifest(), key=lambda e: e.get("received_at", 0))
    if order == "desc":
        entries = list(reversed(entries))

    limit = request.args.get("limit")
    if limit:
        try:
            entries = entries[: int(limit)]
        except ValueError:
            pass  # bad limit value — just ignore it rather than 400 the whole list

    return jsonify({"count": len(entries), "order": order, "videos": entries})


@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No 'video' file in request."}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(video_file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": f"Unsupported video type '{ext}'."}), 400

    mode = request.form.get("mode", "direct")

    # Captured the moment the upload lands — this, not "when processing
    # finished", is what defines "receiving order" for /videos below.
    received_at = time.time()

    # Timestamp-prefixed so concurrent uploads never collide, and so the
    # pipeline's own log filename (derived from the video's basename)
    # stays unique too.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = secure_filename(video_file.filename) or "video.mp4"
    in_name = f"{stamp}_{safe_name}"
    in_path = UPLOAD_DIR / in_name
    video_file.save(str(in_path))

    # This is the "if polygon is available or not" switch: junction
    # zones from the drawn shapes, or [] to fall back to angle-only turn
    # detection when the user ran in direct mode.
    turn_zones = _parse_turn_zones(request.form)
    pipeline.TURN_ZONES = turn_zones

    try:
        pipeline.run_video(str(in_path), str(OUTPUT_DIR))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Pipeline failed: {exc}"}), 500

    # run_video() writes its output as OUTPUT_DIR/<input's basename> —
    # mirror that naming here instead of re-deriving a different one.
    produced_path = OUTPUT_DIR / in_name
    if not produced_path.exists():
        return jsonify({"error": "Pipeline finished but no output video was found."}), 500

    # deepsortVideo.py's logs filename: LOG_DIR/<basename up to first '.'>.txt
    log_path = RUN_LOG_DIR / f"{in_name.split('.')[0]}.txt"
    logs = _parse_log_file(log_path)

    entry = {
        "id": in_name,                                    # stable key, doubles as the saved filename
        "original_filename": video_file.filename,
        "output_video_url": f"/outputs/{produced_path.name}",
        "mode": mode,
        "junction_count": len(turn_zones),
        "received_at": received_at,                        # epoch seconds — sort key for /videos
        "received_at_iso": datetime.fromtimestamp(received_at, tz=timezone.utc).isoformat(),
        "log_row_count": len(logs),
    }
    _append_manifest(entry)

    return jsonify({
        "output_video_url": entry["output_video_url"],
        "logs": logs,
        "mode": mode,
        "junction_count": len(turn_zones),
        "received_at": entry["received_at"],
    })


if __name__ == "__main__":
    from pyngrok import ngrok

    # If you have an ngrok account, set your authtoken first so the
    # tunnel doesn't get killed by the anonymous rate limit:
    #   from pyngrok import conf
    #   conf.get_default().auth_token = "YOUR_NGROK_AUTHTOKEN"

    PORT = 5000
    public_url = ngrok.connect(PORT, "http")
    print(f" * ngrok tunnel:  {public_url}")
    print(f" * Paste this into the frontend's backend-URL field: {public_url}")

    app.run(host="0.0.0.0", port=PORT)

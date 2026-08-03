import time
import subprocess
from pathlib import Path
from collections import deque, defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO
import os;
from torchvision import transforms
from blink_model import build_model, load_weights,ModelConfig
# --- Deep SORT (nwojke/deep_sort) imports ---
# Make sure `deep_sort` repo folder is in PYTHONPATH or installed as package
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort import preprocessing  # NMS on detections before tracker.update
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet  # no longer used for the encoder now that OSNet replaces mars-small128; kept in case you want to fall back to it
from torchreid.utils import FeatureExtractor  # pip install torchreid
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device ="cuda" if torch.cuda.is_available() else "cpu";
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

# ----------------- User params -----------------
VIDEO_PATH = "/content/deepSort/Input";  # apna video path yahan
OUTPUT_PATH = "/content/drive/MyDrive/output-Bike" 
LOG_DIR=  "/content/drive/MyDrive/output-Bike/Logs"   # optional, processed video save karna ho
YOLO_MODEL = "yolov10m.pt"
# Bike-specific OSNet ReID weights (torchreid .pth.tar checkpoint), trained
# on your motorcycle re-identification dataset. Replaces the old TF
# mars-small128.pb pedestrian encoder — see build_osnet_encoder() below for
# the wrapper that makes it a drop-in replacement for gdet.create_box_encoder.
REID_WEIGHTS = "/content/drive/MyDrive/Motor-Bikes-Data/Replace-files/model.pth.tar-50"
OSNET_MODEL_NAME = "osnet_x1_0"          # matches build_model(name='osnet_x1_0', ...) in training
OSNET_INPUT_SIZE = (256, 256)            # (H, W) — matches ImageDataManager(height=256, width=256, ...)
                                          # NOTE: square, not the pedestrian-standard 256x128. Getting
                                          # this wrong silently distorts every crop at inference and
                                          # degrades the embeddings without throwing any error.
# Old combined mirror/indicator model — now used ONLY for the mirror class.
MIRROR_DETECTOR = "/content/deepSort/mirror-indicator-yolov10m/weights/bestM.pt"
# New model: yolov12m, trained on a single class ("indicator") only. Swap
# this path to wherever that weight file (e.g. best.pt) lives on disk.
INDICATOR_DETECTOR = "/content/deepSort/indicator-yolov12m/weights/best.pt"
Orientation_detector_path ='/content/drive/MyDrive/Motor-Bikes-Data/oreintation detector.pt'; 
# Tuned for mars-small128's pedestrian embedding space — OSNet bike features
# live in a different feature space, so re-tune this empirically once the
# new encoder is in (try a validation clip, sweep ~0.15-0.4, pick whichever
# minimizes ID switches without over-merging distinct bikes).
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100
MIN_CONFIDENCE = 0.3
NMS_MAX_OVERLAP = 1.0   # NMS applied to raw detections before they reach the tracker
N_INIT = 3
MAX_AGE = 30
DRAW_TRAILS = True
TRAIL_LEN = 20

# ----------------- Turn zones -----------------
# Define zero, one, or many "turn zones" — regions of the frame where, if a
# bike's box overlaps enough, it's considered to be taking a turn (in
# addition to the angle-based detector in track.py). Any mix of shapes is
# fine, and the list can be empty (pipeline then runs on angle-detection
# alone, same as before).
#
# Each zone is a dict:
#   {"type": "polygon",   "points": [(x1,y1), (x2,y2), (x3,y3), ...]}   # >=3 points, any shape
#   {"type": "rectangle", "points": [(x1,y1), (x2,y2)]}                  # top-left, bottom-right corners
#                                                                        # (a 4-point list also works — treated as a polygon)
#   {"type": "circle",    "center": (cx, cy), "radius": r}
#
# Example (uncomment / edit for your video):
# TURN_ZONES = [
#     {"type": "polygon",   "points": [(120, 300), (420, 300), (500, 520), (60, 520)]},
#     {"type": "rectangle", "points": [(900, 150), (1180, 420)]},
#     {"type": "circle",    "center": (1500, 650), "radius": 140},
# ]
TURN_ZONES = []

# % (0.0-1.0) of a bike's bounding-box area that must fall inside a turn
# zone before it counts as "taking the turn". Kept as its own variable so it
# can be tuned later without touching the detection logic itself.
TURN_ZONE_OVERLAP_THRESHOLD = 0.30

# Color/thickness the zones themselves are drawn with in the output video.
TURN_ZONE_DRAW_COLOR = (255, 255, 0)   # BGR — cyan-ish, distinct from bike/mirror/indicator boxes
TURN_ZONE_DRAW_THICKNESS = 2
# ------------------------------------------------
counter = 0;
os.makedirs(OUTPUT_PATH , exist_ok=True);
os.makedirs(LOG_DIR , exist_ok=True);


#----------------------------------------------------

cfg = ModelConfig(cnn_name="resnet18", seq_type="LSTM", num_layers=2, hidden_size=256, dropout=0.5)
blink_detector = build_model(cfg);
blink_detector = load_weights(blink_detector, "/content/drive/MyDrive/Motor-Bikes-Data/blink model weights/best_model.pth", device=device) 


# ImageNet normalization (resnet18 pretrained isi se train hua tha)
_normalize = transforms.Compose([
    transforms.ToTensor(),   # (H, W, C) numpy -> (C, H, W) tensor, scaled to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict_from_track(model, track, slot_idx, device="cpu"):
    """
    Track's stored 64x64 indicator images (for ONE indicator slot — left or
    right, kept separate) -> single blink prediction for that slot.
    Returns (predicted_class, confidence) or None if that slot has no
    stored crops yet.
    """
    frames = track.getIndicatorImages(slot_idx)
    if not frames:
        return None

    processed = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        chw = _normalize(frame)          # (C, 64, 64), normalized
        hwc = chw.permute(1, 2, 0)       # (64, 64, C) -- model ye layout expect karta hai
        processed.append(hwc)

    seq = torch.stack(processed, dim=0).unsqueeze(0).to(device)   # (1, T, 64, 64, C)
    length = torch.tensor([seq.shape[1]], device=device)

    model.eval()
    logits = model(seq, length)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0, pred].item()

    return pred, conf


def extract_indicator_crop(frame, indicator_box, size=(64, 64)):
    """
    Crop indicator from frame, resize to 64x64 and return numpy array.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    indicator_box : list/tuple
        [confidence, x1, y1, x2, y2]
        or [x1, y1, x2, y2]
    size : tuple
        Output image size (default 64x64)

    Returns
    -------
    np.ndarray or None
        Shape: (64, 64, 3)
    """

    # Handle both formats
    if len(indicator_box) == 5:
        _, x1, y1, x2, y2 = indicator_box
    else:
        x1, y1, x2, y2 = indicator_box

    h, w = frame.shape[:2]

    # Clip coordinates
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # blink model expects RGB, frame is BGR (OpenCV)

    return crop.astype(np.uint8)


def build_osnet_encoder(model_path, model_name=OSNET_MODEL_NAME,
                         input_size=OSNET_INPUT_SIZE, device="cpu"):
    """
    Returns a callable `encoder(frame, boxes_tlwh) -> np.ndarray (N, feat_dim)`
    that matches the exact interface gdet.create_box_encoder(...) used to
    provide, so it drops straight into `features = encoder(frame, tlwhs)`
    with no other changes needed in run_video().

    Uses torchreid's FeatureExtractor, which handles resizing/normalization
    to match how the model was trained — just make sure `image_size` here
    matches what you trained OSNet at, or crops get distorted.
    """
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        image_size=input_size,   # (H, W)
        device=device,
    )

    def encoder(frame, boxes_tlwh):
        if len(boxes_tlwh) == 0:
            return np.zeros((0, 512), dtype=np.float32)  # osnet_x1_0 default feat dim

        h_frame, w_frame = frame.shape[:2]
        crops = []
        for (x, y, w, h) in boxes_tlwh:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_frame, int(x + w)), min(h_frame, int(y + h))
            if x2 <= x1 or y2 <= y1:
                # degenerate box (can happen if a tlwh comes in clipped to
                # nothing) — feed a black patch so the batch stays aligned
                # with boxes_tlwh instead of silently dropping an index
                crop = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
            else:
                crop = frame[y1:y2, x1:x2]
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            feats = extractor(crops)                       # (N, feat_dim) torch tensor
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

        return feats.cpu().numpy().astype(np.float32)

    return encoder


# ----------------- Optical flow (deep-learning flow-net) -----------------
# True FlowNet2 (flownet2-pytorch) needs a custom CUDA "correlation" layer
# that has to be compiled from source — a real pain on Colab and it breaks
# every time the CUDA/torch version shifts. RAFT is the modern replacement:
# same idea (a CNN that predicts a dense optical-flow field between two
# frames), pretrained weights ship inside torchvision, no custom ops to
# build. Using it here as the "flow-net". If FlowNet2 is a hard requirement
# (e.g. matching a paper's exact architecture) say so and this can be
# swapped for the flownet2-pytorch build instead — this function is the
# only thing that needs to change, everything downstream just consumes a
# (H, W, 2) flow array.
def build_flow_model(device="cpu"):
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=False).to(device)
    model.eval()
    return model, weights.transforms()


@torch.no_grad()
def compute_optical_flow(flow_model, flow_transforms, prev_frame, curr_frame, device="cpu"):
    """
    prev_frame, curr_frame : BGR numpy arrays at the video's native resolution.
    Returns a (H, W, 2) numpy array — [...,0] = dx, [...,1] = dy per pixel,
    resized back to the ORIGINAL frame resolution so bbox lookups below don't
    need to know anything about RAFT's internal resize. None on the first
    frame (no previous frame to compare against yet).
    """
    if prev_frame is None:
        return None

    orig_h, orig_w = curr_frame.shape[:2]

    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

    prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
    curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)
    prev_t, curr_t = flow_transforms(prev_t, curr_t)
    prev_t, curr_t = prev_t.to(device), curr_t.to(device)

    flow_preds = flow_model(prev_t, curr_t)
    flow = flow_preds[-1]                              # last refinement iter, (1, 2, H', W')
    flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    flow = flow[0].permute(1, 2, 0).cpu().numpy()       # (H, W, 2)
    return flow


def bike_flow_stats(flow, bbox):
    """
    Average optical-flow vector inside a single bike's bbox for this frame.
    Returns (dx, dy, magnitude, angle_deg) — all zero if flow is unavailable
    (first frame) or the box is degenerate.
    """
    if flow is None:
        return 0.0, 0.0, 0.0, 0.0

    h, w = flow.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0

    patch = flow[y1:y2, x1:x2]
    dx = float(np.mean(patch[..., 0]))
    dy = float(np.mean(patch[..., 1]))
    magnitude = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return dx, dy, magnitude, angle


def xyxy_to_tlwh(box):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [int(x1), int(y1), int(w), int(h)]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def is_box_inside(box1, box2, method="center"):
    """
    Check if box2 is inside box1.

    Parameters:
        box1: list or tuple [x1, y1, x2, y2]  # parent box
        box2: list or tuple [x1, y1, x2, y2]  # child box
        method: "center" (default), "full", or "iou"
    
    Returns:
        True if box2 is considered inside box1, else False
    """

    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    if method == "center":
        # Check center point of box2
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        return x1 <= cx <= x2 and y1 <= cy <= y2

    elif method == "full":
        # Check full containment
        return x1 <= bx1 and y1 <= by1 and x2 >= bx2 and y2 >= by2

    elif method == "iou":
        # Check IoU threshold (partial overlap)
        inter_x1 = max(x1, bx1)
        inter_y1 = max(y1, by1)
        inter_x2 = min(x2, bx2)
        inter_y2 = min(y2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (bx2 - bx1) * (by2 - by1)
        iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

        return iou > 0.1  # threshold adjustable

    else:
        raise ValueError("Method must be 'center', 'full', or 'iou'")

def _zone_polygon_points(zone):
    """Return an (N,2) int32 array of vertices for a polygon/rectangle zone.
    Rectangles given as 2 corner points get expanded to 4; anything else
    (>=3 points, arbitrary polygon) passes through as-is. Not used for
    circles, which are rasterized directly.
    """
    pts = zone["points"]
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return np.array(pts, dtype=np.int32)


def build_zone_mask(zones, width, height):
    """Rasterize every configured turn zone — any mix of polygon, rectangle,
    or circle, any count — into a single binary (H, W) mask where a pixel is
    1 if it falls inside ANY zone. Built once per video (zones are treated
    as static for the whole clip), so per-frame overlap checks below are
    just a cheap crop + count against this mask.

    Returns None if `zones` is empty, so callers can skip zone-based turn
    detection entirely when no zones are configured (the "no polygon" case).
    """
    if not zones:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone)
            cv2.fillPoly(mask, [pts_arr], 1)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")
    return mask


def zone_overlap_ratio(mask, bbox):
    """Fraction (0.0-1.0) of `bbox` [x1,y1,x2,y2] that falls inside ANY
    configured turn zone, per the mask from build_zone_mask(). Returns 0.0
    if there's no mask (no zones configured) or the box is degenerate/fully
    off-frame.
    """
    if mask is None:
        return 0.0

    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    box_area = (x2 - x1) * (y2 - y1)
    inside = int(np.count_nonzero(mask[y1:y2, x1:x2]))
    return inside / box_area if box_area > 0 else 0.0


def draw_zones(frame, zones, color=TURN_ZONE_DRAW_COLOR, thickness=TURN_ZONE_DRAW_THICKNESS):
    """Draw every configured turn zone onto `frame` — any mix of polygon,
    rectangle, or circle, any count — so the output video shows exactly what
    bikes are being checked against. No-op if `zones` is empty.
    """
    for zone in zones:
        ztype = zone.get("type", "polygon").lower()
        if ztype == "circle":
            cx, cy = map(int, zone["center"])
            r = int(zone["radius"])
            cv2.circle(frame, (cx, cy), r, color, thickness)
        elif ztype in ("polygon", "rectangle"):
            pts_arr = _zone_polygon_points(zone).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=thickness)
        else:
            raise ValueError(f"Unknown turn zone type: {ztype!r} (expected 'polygon', 'rectangle', or 'circle')")


def performOnVideo(track, frame, trails, mirror_indicator_detector):
    global counter
    track_id = track.track_id
    bbox = track.to_tlbr() 
    x1, y1, x2, y2 = map(int, bbox)
    basepath =f'/photostos/{track_id}' 
    os.makedirs(basepath, exist_ok=True)
    motorcycle_crop = frame[y1:y2, x1:x2].copy()
    
    cv2.imwrite(f"{basepath}/Out _ {counter} _ {track_id}.jpg",motorcycle_crop )
    counter+=1;

    mirror_results = mirror_indicator_detector(motorcycle_crop)
    res = mirror_results[0]

    track.clearCords();

    # Update track mask/glasses status
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = res.names[cls_id]
        mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
        mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
        global_x1 = int(x1 + mx1)
        global_y1 = int(y1 + my1)
        global_x2 = int(x1 + mx2)
        global_y2 = int(y1 + my2)

        if "mirror" in cls_name:
            # Track.setMirror(self, cord) takes ONE arg (the coords list).
            # This used to call setMirror(cls_name, [...]) with two args,
            # which would raise a TypeError the moment this dead code path
            # was ever actually invoked. Matching the single-arg signature
            # used everywhere else (run_video's bike.setMirror(mirror) call).
            track.setMirror([global_x1, global_y1, global_x2, global_y2])
        #if "glass" in cls_name or "no_glass" in cls_name:
            #track.setGlasses(cls_name, [ global_x1 ,global_y1 ,global_x2,global_y2 ])

    # Draw mask
    if track.mirror_Cord is not None and len(track.mirror_Cord) > 0:
        drawn_boxes = []  # already drawn boxes store karne ke liye
        for cords in track.mirror_Cord[-1::-1]:  # last element ya aap jitne chahen

            mx1, my1, mx2, my2 = map(int, cords)
            new_box = (mx1, my1, mx2, my2)
            
            skip = False
            for box in drawn_boxes: 
                if compute_iou(new_box, box) > 0:
                    skip = True
                    break
            
            if skip:
                continue
            
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            cv2.putText(frame, 'Mirror', (mx1, my1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            drawn_boxes.append(new_box) 


    # Draw glasses
    #if track.Glasses_Cord is not None and len(track.Glasses_Cord) > 0:
        #gx1, gy1, gx2, gy2 = map(int, track.Glasses_Cord)
        #cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        #cv2.putText(frame, track.Glasses_status, (gx1, gy1-6),
                  #  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw track bounding box + summary
    color = ((track_id * 37) % 255, (track_id * 17) % 255, (track_id * 29) % 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, track.GetSummary(), (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw center point & trail
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    if DRAW_TRAILS:
        if track_id not in trails:
            trails[track_id] = deque(maxlen=TRAIL_LEN)
        trails[track_id].appendleft((cx, cy))
        pts = list(trails[track_id])
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j-1], pts[j], color, 2)


def expand_box(x1,y1,x2,y2,  w, h, pad_ratio=0.30):
    bw = x2-x1
    bh = y2-y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    x1 = max(0, x1-pad_w)
    y1 = max(0, y1-pad_h)
    x2 = min(w, x2+pad_w)
    y2 = min(h, y2+pad_h)

    return x1,y1,x2,y2


def getDetections(model, frame , detect , confidence = MIN_CONFIDENCE):
        results =model(frame , imgsz = 1280)
        detections = []
        res =results[0]
        class_name =[]
        if hasattr(res, "boxes"):
             
            for box in res.boxes:
                cls_id= int(box.cls[0]) if hasattr(box.cls, "__getitem__") else int(box.cls)
                name = res.names[cls_id]
                print(f'ClassName : give: {detect} -> found {name}')
                mx1, my1, mx2, my2 = box.xyxy[0].cpu().numpy()
                mx1, my1, mx2, my2 = int(mx1), int(my1), int(mx2), int(my2)
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf, "__getitem__") else float(box.conf)               
                if conf > confidence and name in detect:
                    detections.append([conf, mx1, my1, mx2, my2])
                    class_name.append(name)
                
        return detections , class_name


def active_tracks(tracker):
    """
    Tracks that are confirmed (passed n_init) AND were actually matched to a
    detection this frame (time_since_update == 0).

    Iterating tracker.tracks directly (as the original code did) also pulls
    in tentative tracks (unconfirmed, can vanish next frame) and stale/ghost
    tracks that Deep SORT is still coasting on Kalman prediction alone for up
    to max_age frames after last seeing them. Drawing/associating against
    those is what produces "duplicate" boxes on a single physical object
    when a fresh track spawns for it while the old ghost track is still
    being rendered.
    """
    return [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]


def best_match_track(tracks, box, method="center"):
    """
    Return the single track whose box has the highest IoU with `box` among
    those where is_box_inside(...) also holds, or None.

    The original code assigned a mirror/indicator/orientation detection to
    EVERY track that satisfied is_box_inside(). When two tracked bikes
    overlap (common in traffic), that lets one detection get attached to
    more than one track -> duplicate annotations that look like ID mixups.
    Picking a single best match fixes that.
    """
    best_track, best_iou = None, 0.0
    for t in tracks:
        t_box = t.to_tlbr()
        if not is_box_inside(t_box, box, method=method):
            continue
        iou = compute_iou(t_box, box)
        if iou >= best_iou:
            best_iou = iou
            best_track = t
    return best_track


def draw_box(image, coords, label="Box", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def run_video(video_path, output_path=None):
    vidName = video_path.split("/")[-1];
    if output_path != None:
        output_path = os.path.join(output_path ,  vidName);
    else:
        output_path = "output.mp4";

    logs = os.path.join(LOG_DIR,vidName.split(".")[0]+".txt");
    yolo = YOLO(YOLO_MODEL).to(device)
    mirror_detector = YOLO(MIRROR_DETECTOR).to(device)
    indicator_detector = YOLO(INDICATOR_DETECTOR).to(device)
    orientation_detector = YOLO(Orientation_detector_path).to(device)
    flow_model, flow_transforms = build_flow_model(device)

    # Init DeepSORT
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=MAX_AGE, n_init=N_INIT)

    encoder = build_osnet_encoder(REID_WEIGHTS, device=device)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # OpenCV's mp4v encoder is reliable to write to, but the resulting file
    # uses a codec browsers can't decode (only desktop players like VLC/WMP
    # can). So we write to a temp file with mp4v as before, then re-encode
    # it to H.264 (browser-playable) as a final step below, and delete the
    # temp file. `output_path` stays the final, browser-compatible filename.
    raw_output_path = None
    writer = None
    if output_path:
        raw_output_path = output_path.replace(".mp4", "_raw_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Built once — zones are static for the whole clip. None if TURN_ZONES
    # is empty, in which case zone_overlap_ratio() always returns 0.0 and
    # the pipeline runs on angle-based turn detection alone (unchanged
    # behavior for videos with no zones configured).
    zone_mask = build_zone_mask(TURN_ZONES, width, height)

    trails = {}
    frame_idx = 0
    prev_frame = None
    t0 = time.time()

    # CSV header written once, upfront — everything below is appended per
    # frame. human_verification defaults to "not turn" for every row; that
    # column is meant to be hand-corrected later while reviewing the video,
    # not touched by the pipeline itself.
    with open(logs, "w", encoding="utf-8") as f:
        f.write(
            "frame_idx,track_id,cx,cy,orientation_side,orientation_frontback,"
            "flow_dx,flow_dy,flow_magnitude,flow_angle,cum_angle_change,"
            "zone_overlap_ratio,is_turning_algo,human_verification\n"
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        flow = compute_optical_flow(flow_model, flow_transforms, prev_frame, frame, device=device)

        motorcycle_boxes ,_ = getDetections(yolo , frame , ['motorcycle']);
        detections = []
        if motorcycle_boxes:
            tlwhs = [xyxy_to_tlwh(p[1:]) for p in motorcycle_boxes]
            confidences = [p[0] for p in motorcycle_boxes]
            features = encoder(frame, tlwhs)

            # --- NMS on raw detections BEFORE they reach the tracker -------
            # Without this, two overlapping YOLO boxes on the same bike both
            # get fed to tracker.update() and Deep SORT can spawn a second
            # track for the same physical object -> duplicate IDs.
            boxes_arr = np.array(tlwhs, dtype=np.float32)
            scores_arr = np.array(confidences, dtype=np.float32)
            keep = preprocessing.non_max_suppression(boxes_arr, NMS_MAX_OVERLAP, scores_arr)

            for i in keep:
                detections.append(Detection(tlwhs[i], confidences[i], features[i]))

        tracker.predict()
        tracker.update(detections)

        mirrors,_  = getDetections(mirror_detector, frame , ['mirror']);
        indicators,_  = getDetections(indicator_detector, frame , ['indicator']);
        orientation_boxes, orientation_classes = getDetections(
            orientation_detector,
            frame,
            ['front-back', 'side']
        )

        for i in tracker.tracks:
            i.clearCords();

        # Only associate/draw against tracks that are confirmed AND were
        # actually matched to a detection this frame. Using tracker.tracks
        # directly pulls in tentative and stale/ghost tracks, which is the
        # other main source of "duplicate" boxes and mis-assigned
        # mirror/indicator/orientation detections.
        live_tracks = active_tracks(tracker)

        # Update trajectories / turn detection FIRST, so bike.is_turning
        # below reflects THIS frame rather than lagging by one frame (as in
        # the original ordering, where detect_turn() ran after is_turning
        # was already consumed for the indicator/blink check).
        turn_results = {}
        for bike in live_tracks:
            bike.setTrajectories(bike.to_tlbr())
            turn_results[bike.track_id] = bike.detect_turn()

            # Turn-zone check: works alongside the angle-based detector
            # above, not instead of it — either one flagging a turn is
            # enough. With TURN_ZONES empty, zone_mask is None and this is a
            # no-op (ratio 0.0, is_turning left exactly as detect_turn() set
            # it), so this is safe to leave on for every video.
            ratio = zone_overlap_ratio(zone_mask, bike.to_tlbr())
            bike.zone_overlap_ratio = ratio
            if ratio >= TURN_ZONE_OVERLAP_THRESHOLD:
                bike.is_turning = True

        for box, orientation in zip(orientation_boxes, orientation_classes):
            bbox = box[1:]  # x1, y1, x2, y2
            bike = best_match_track(live_tracks, bbox, method="iou")
            if bike is not None:
                bike.setOrientation(orientation)

        for mirror in mirrors:
            bike = best_match_track(live_tracks, mirror[1:], method="center")
            if bike is not None:
                bike.setMirror(mirror)

        # Group this frame's indicator detections by the track they matched,
        # so both indicators on a bike (0, 1, or 2 boxes) are handed to
        # update_indicators() together — the nearest-neighbor slot matching
        # needs to see them as a set, not one at a time, or it can't tell
        # which detection is "the one that was already left" vs "right".
        indicators_by_track = defaultdict(list)
        for indicator in indicators:
            bike = best_match_track(live_tracks, indicator[1:], method="center")
            if bike is not None:
                indicators_by_track[bike.track_id].append(indicator)

        for bike in live_tracks:
            dets = indicators_by_track.get(bike.track_id, [])

            # How far the bike itself moved this frame, so a slot that
            # misses detection this frame can still shift forward with the
            # bike instead of freezing or vanishing.
            traj = bike.getAllTrajectoriesOfX()
            if len(traj) >= 2:
                move_dx = traj[-1][0] - traj[-2][0]
                move_dy = traj[-1][1] - traj[-2][1]
            else:
                move_dx, move_dy = 0.0, 0.0

            bike.update_indicators(dets, move_dx, move_dy)

            if not bike.is_turning:
                continue

            any_signal = False
            for slot_idx, slot in enumerate(bike.indicator_slots):
                if slot["cord"] is None:
                    continue

                croped = extract_indicator_crop(frame, slot["cord"])
                if croped is not None:
                    # Each slot keeps its own crop history, so the blink
                    # model is trained/queried per-indicator, not on a mix
                    # of left+right frames.
                    bike.setIndicatorImage(slot_idx, croped)

                result = predict_from_track(blink_detector, bike, slot_idx, device=device)
                if result:
                    pred_class, confidence = result
                    slot["is_signal"] = True if pred_class == 0 else False
                    print(f"Track {bike.track_id} slot {slot_idx} prediction: {pred_class} ({confidence:.2%})")
                any_signal = any_signal or slot["is_signal"]

            # Bike-level flag: on if EITHER indicator is judged to be
            # blinking (normal turning uses just one side; both only for
            # hazards, which still counts as "signaling").
            bike.is_signal = any_signal

        # Centralized color rule — applies to EVERY live track (not just
        # the ones that were turning), so straight bikes get green too.
        # Same rule regardless of whether is_turning came from the
        # angle-based detector, a turn-zone overlap, or both:
        #   not turning            -> green
        #   turning, no indicator  -> red
        #   turning, indicator ON  -> black
        for bike in live_tracks:
            bike.update_box_color()

        print('================= Draw State ==============================')
        # No-op if TURN_ZONES is empty — safe to always call.
        draw_zones(frame, TURN_ZONES)

        log_rows = []
        for bikes in live_tracks:
            bbox = bikes.to_tlbr()
            strs = "N"

            if bikes.is_turning and bikes.is_signal:
                strs = "TO"
            if bikes.is_turning and not bikes.is_signal:
                strs = "TF"

            x1, y1, x2, y2 = map(int, bbox)
            _, cum_angle, _ = turn_results.get(bikes.track_id, (False, 0.0, ""))

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            flow_dx, flow_dy, flow_mag, flow_angle = bike_flow_stats(flow, bbox)
            bikes.setFlow((flow_dx, flow_dy, flow_mag, flow_angle))

            log_rows.append(
                f"{frame_idx},{bikes.track_id},{cx:.1f},{cy:.1f},"
                f"{bikes.orientation['side']},{bikes.orientation['front-back']},"
                f"{flow_dx:.3f},{flow_dy:.3f},{flow_mag:.3f},{flow_angle:.2f},"
                f"{cum_angle:.2f},{bikes.zone_overlap_ratio:.3f},{bikes.is_turning},not turn"
            )

            draw_box(frame, (x1, y1, x2, y2), f'bike: {bikes.track_id} - {strs}', bikes.box_color)
            for mirror in bikes.mirror_Cord:
                draw_box(frame, mirror[1:], f'Mirror: {bikes.track_id}')
            for slot_idx, slot in enumerate(bikes.indicator_slots):
                if slot["cord"] is None:
                    continue
                label = f'Ind-{slot_idx} {bikes.track_id}' + (' ON' if slot["is_signal"] else '')
                slot_color = (0, 255, 0) if slot["is_signal"] else (0, 165, 255)
                draw_box(frame, slot["cord"][1:], label, slot_color)

        if log_rows:
            with open(logs, "a", encoding="utf-8") as f:
                f.write("\n".join(log_rows) + "\n")

        if writer:
            writer.write(frame)
        #cv2.imshow("Video Tracking", frame)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

        prev_frame = frame.copy()

    cap.release()
    if writer:
        writer.release()
    #cv2.destroyAllWindows()

    # Re-encode the mp4v temp file to H.264 so it plays in browsers
    # (mp4v only plays in desktop players like VLC/WMP, not <video> tags).
    # -movflags +faststart moves metadata to the front of the file so it
    # can start streaming/playing before it's fully downloaded.
    if raw_output_path and os.path.exists(raw_output_path):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", raw_output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    output_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.remove(raw_output_path)
            print(f"Saved browser-compatible video: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # ffmpeg missing or conversion failed -- keep the raw mp4v file
            # around (rename it back to the expected output_path) instead of
            # silently losing the processed video.
            print(f"WARNING: ffmpeg re-encode failed ({e}); "
                  f"keeping original mp4v file (plays in VLC/media players, not browsers).")
            os.replace(raw_output_path, output_path)


if __name__ == "__main__":

    for vid in os.listdir(VIDEO_PATH):
        run_video(os.path.join(VIDEO_PATH , vid), OUTPUT_PATH) 


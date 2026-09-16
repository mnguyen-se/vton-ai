"""
pose_detect.py
---------------
Auto-detects body keypoints (shoulders, hips, knees, ankles) directly on a
SINGLE photo using MediaPipe Pose - no matching "bare" reference photo
needed, and no manual clicking needed either.

WHY THIS EXISTS
Phase 2's original prepare_dataset.py built the inpaint mask by diffing
mannequin_bare.jpg against mannequin_wearing.jpg pixel-by-pixel. That only
works if both photos are the SAME mannequin, SAME pose, SAME camera/crop -
literally a before/after pair. It silently produces garbage masks when fed
real-world data: e-commerce product photos of different garments are almost
always different photoshoots (different mannequin/model, pose, crop, aspect
ratio), even when a single generic "bare" reference photo is reused across
all of them. Detecting pose directly on each wearing photo sidesteps the
whole pairing requirement.

Coordinates are returned as FRACTIONS of the image's own width/height
(0.0-1.0), same convention as pipeline/mannequin_pose.py's saved keypoints,
so a box built here can be scaled to any target resolution consistently.

MEDIAPIPE API NOTE: recent mediapipe releases (>= 0.10.x on a fresh `pip
install mediapipe`) removed the old `mp.solutions.pose.Pose` API entirely
in favor of the Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`),
which needs a small model file downloaded once. This module uses the Tasks
API and auto-downloads that file to a local cache dir on first use (needs
internet the first time only). If you've pinned an older mediapipe that
still ships `mp.solutions`, this module falls back to that automatically.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import urllib.request
from PIL import Image
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

# Landmark indices are the same in both the legacy and Tasks APIs (BlazePose's
# standard 33-point topology).
_NOSE = 0
_L_EAR, _R_EAR = 7, 8
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28
_L_HEEL, _R_HEEL = 29, 30
_L_FOOT_INDEX, _R_FOOT_INDEX = 31, 32

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
_MODEL_PATH = Path.home() / ".cache" / "vton_pipeline" / "pose_landmarker_full.task"


@dataclass
class DetectedKeypoints:
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    left_hip: tuple[float, float]
    right_hip: tuple[float, float]
    left_knee: tuple[float, float]
    right_knee: tuple[float, float]
    left_ankle: tuple[float, float]
    right_ankle: tuple[float, float]
    nose: tuple[float, float]
    left_ear: tuple[float, float]
    right_ear: tuple[float, float]
    left_heel: tuple[float, float]
    right_heel: tuple[float, float]
    left_foot_index: tuple[float, float]
    right_foot_index: tuple[float, float]
    confidence: float  # mean visibility score across the core points, 0-1

    def torso_box(self) -> tuple[float, float, float, float]:
        """Fractional (x0,y0,x1,y1) box for a top garment: shoulders to hips."""
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0]]
        ys = [self.left_shoulder[1], self.right_shoulder[1], self.left_hip[1], self.right_hip[1]]
        return min(xs), min(ys), max(xs), max(ys)

    def lower_body_box(self) -> tuple[float, float, float, float]:
        """Fractional (x0,y0,x1,y1) box for a bottom garment: hips to ankles."""
        xs = [self.left_hip[0], self.right_hip[0], self.left_ankle[0], self.right_ankle[0]]
        y0 = min(self.left_hip[1], self.right_hip[1])
        y1 = max(self.left_ankle[1], self.right_ankle[1])
        return min(xs), y0, max(xs), y1

    def head_box(self) -> tuple[float, float, float, float]:
        """
        Fractional (x0,y0,x1,y1) box for a hat: MediaPipe has no top-of-head
        landmark, so we estimate it from the nose/ear/shoulder geometry -
        head width from ear-to-ear (widened a bit), head top extrapolated
        upward from nose by the typical nose-to-shoulder distance fraction
        that corresponds to a head's height. This is an approximation, not
        exact - hat photos tend to be forgiving of a slightly loose box
        since a hat's own silhouette dominates what gets inpainted anyway.
        """
        ear_xs = [self.left_ear[0], self.right_ear[0]]
        shoulder_y = (self.left_shoulder[1] + self.right_shoulder[1]) / 2
        head_width = abs(ear_xs[1] - ear_xs[0])
        head_width = max(head_width, 0.05)  # guard against near-zero (near-profile) detections
        x0, x1 = min(ear_xs) - head_width * 0.3, max(ear_xs) + head_width * 0.3
        # Head height is roughly the nose-to-shoulder distance; top-of-head
        # sits a bit above the nose by about that same distance.
        head_height = max(shoulder_y - self.nose[1], head_width)
        y0 = self.nose[1] - head_height * 0.9
        y1 = self.nose[1] + head_height * 0.35
        return x0, max(0.0, y0), x1, y1

    def feet_box(self) -> tuple[float, float, float, float]:
        """Fractional (x0,y0,x1,y1) box for shoes: ankle to heel/toe, both feet."""
        xs = [self.left_heel[0], self.right_heel[0], self.left_foot_index[0], self.right_foot_index[0],
              self.left_ankle[0], self.right_ankle[0]]
        ys = [self.left_heel[1], self.right_heel[1], self.left_foot_index[1], self.right_foot_index[1],
              self.left_ankle[1], self.right_ankle[1]]
        y0 = min(self.left_ankle[1], self.right_ankle[1])  # top of box starts at ankle, not higher
        return min(xs), y0, max(xs), max(ys)

    def dress_box(self) -> tuple[float, float, float, float]:
        """
        Fractional (x0,y0,x1,y1) box for a one-piece dress: shoulders down
        to near the ankles - the union of what torso_box() and
        lower_body_box() cover separately for a top+bottom pair. A dress
        is ONE item covering both regions, so it gets its own box rather
        than requiring the caller to combine top+bottom (which would also
        wrongly imply two separate garments are being applied).
        """
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0],
              self.left_knee[0], self.right_knee[0]]
        y0 = min(self.left_shoulder[1], self.right_shoulder[1])
        y1 = max(self.left_ankle[1], self.right_ankle[1])
        return min(xs), y0, max(xs), y1


def _ensure_model_downloaded() -> str:
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose_detect] Downloading pose model (~30MB, one-time) from {_MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the MediaPipe pose model to {_MODEL_PATH}: {e}\n"
            "This needs internet access on first use. If this machine is offline/firewalled, "
            "download the file manually from the URL above on another machine and place it at "
            f"{_MODEL_PATH}."
        ) from e
    return str(_MODEL_PATH)


_detector_singleton = None
_use_legacy_api = None


def _get_detector():
    """Returns either a Tasks-API PoseLandmarker or a legacy solutions.Pose object."""
    global _detector_singleton, _use_legacy_api
    if _detector_singleton is not None:
        return _detector_singleton

    if not _HAS_MEDIAPIPE:
        raise ImportError(
            "mediapipe is required for auto pose detection. Install it with:\n"
            "    pip install mediapipe\n"
            "(it's already in this project's install instructions)."
        )

    if hasattr(mp, "solutions"):
        # Older mediapipe (pinned) - still has the simple legacy API.
        _use_legacy_api = True
        _detector_singleton = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3,
        )
    else:
        # Recent mediapipe - solutions.pose was removed, use the Tasks API.
        _use_legacy_api = False
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        model_path = _ensure_model_downloaded()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
        )
        _detector_singleton = vision.PoseLandmarker.create_from_options(options)

    return _detector_singleton


def detect_keypoints(image: Image.Image) -> DetectedKeypoints | None:
    """
    Returns DetectedKeypoints with fractional (0-1) coordinates, or None if
    no person/mannequin could be confidently detected in the image (e.g. a
    headless "ghost mannequin" product photo, or a very cropped/occluded shot).
    """
    detector = _get_detector()
    rgb = np.array(image.convert("RGB"))

    if _use_legacy_api:
        result = detector.process(rgb)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks.landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]  # first (only, since num_poses=1) detected person

    def pt(i):
        lm = landmarks[i]
        return (lm.x, lm.y), lm.visibility

    ls, ls_v = pt(_L_SHOULDER); rs, rs_v = pt(_R_SHOULDER)
    lh, lh_v = pt(_L_HIP); rh, rh_v = pt(_R_HIP)
    lk, lk_v = pt(_L_KNEE); rk, rk_v = pt(_R_KNEE)
    la, la_v = pt(_L_ANKLE); ra, ra_v = pt(_R_ANKLE)
    nose, nose_v = pt(_NOSE)
    le, le_v = pt(_L_EAR); re, re_v = pt(_R_EAR)
    lheel, lheel_v = pt(_L_HEEL); rheel, rheel_v = pt(_R_HEEL)
    lfi, lfi_v = pt(_L_FOOT_INDEX); rfi, rfi_v = pt(_R_FOOT_INDEX)
    # Confidence is averaged over the core torso/leg points only - hat/shoe
    # boxes are approximations anyway (see head_box docstring), so a
    # momentary low-visibility ear or heel point (profile angle, cropped
    # frame) shouldn't tank the overall confidence score used to accept/
    # reject the whole detection.
    confidence = float(np.mean([ls_v, rs_v, lh_v, rh_v, lk_v, rk_v, la_v, ra_v]))

    return DetectedKeypoints(
        left_shoulder=ls, right_shoulder=rs,
        left_hip=lh, right_hip=rh,
        left_knee=lk, right_knee=rk,
        left_ankle=la, right_ankle=ra,
        nose=nose, left_ear=le, right_ear=re,
        left_heel=lheel, right_heel=rheel,
        left_foot_index=lfi, right_foot_index=rfi,
        confidence=confidence,
    )
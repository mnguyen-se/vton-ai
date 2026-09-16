"""
mannequin_pose.py
------------------
Detects body keypoints on the mannequin photo so we know where to place
and warp the garment (shoulders, waist, hips, etc).

IMPORTANT CAVEAT:
MediaPipe Pose is trained on real human photos, not mannequins. On plain
plastic/fabric mannequins (no face, stylised proportions) accuracy can be
inconsistent. Two supported modes:

  1. AUTO   - MediaPipe PoseLandmarker (works reasonably on realistic /
              "ghost mannequin" style photos, less reliable on abstract
              headless mannequins).
  2. MANUAL - You click/annotate keypoints once per mannequin photo
              (recommended for a small fixed set of mannequin poses,
              since you likely reuse the same few mannequin shots a lot).

For a production system with a limited number of mannequin base photos,
MANUAL calibration (done once per mannequin image) is more reliable than
AUTO detection and is what we recommend as the default.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class BodyKeypoints:
    left_shoulder: tuple[float, float]
    right_shoulder: tuple[float, float]
    left_hip: tuple[float, float]
    right_hip: tuple[float, float]
    left_knee: tuple[float, float] | None = None
    right_knee: tuple[float, float] | None = None
    neck: tuple[float, float] | None = None

    def torso_box(self) -> tuple[float, float, float, float]:
        """
        Fractional (0-1) bounding box for the torso region (top garment).
        NOTE: stays in the same 0-1 fractional space as the stored
        keypoints - do NOT cast to int here. This object stores fractions
        of the calibration photo's width/height (see to_pixels()), and an
        int() cast here would truncate every coordinate to 0.
        """
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0]]
        ys = [self.left_shoulder[1], self.right_shoulder[1], self.left_hip[1], self.right_hip[1]]
        return min(xs), min(ys), max(xs), max(ys)

    def lower_body_box(self, image_height: float = 1.0) -> tuple[float, float, float, float]:
        """
        Fractional (0-1) bounding box for hips-to-ankle region (bottom garment).
        image_height: pass 1.0 (the default) to get a fractional box; only
        pass an actual pixel height if you specifically want this one box
        pre-scaled to pixels while everything else stays fractional (not
        recommended - prefer scaling everything at once via to_pixels()).
        No manual pixel padding is added here (the old hardcoded +-20px was
        a pixel-space assumption that silently broke once coordinates
        became fractional); callers add their own padding as a fraction of
        box size, same as pose_detect.DetectedKeypoints.lower_body_box().
        """
        xs = [self.left_hip[0], self.right_hip[0]]
        y0 = min(self.left_hip[1], self.right_hip[1])
        y1 = image_height * 0.98
        return min(xs), y0, max(xs), y1

    def dress_box(self, image_height: float = 1.0) -> tuple[float, float, float, float]:
        """
        Fractional (0-1) box for a one-piece dress: shoulders down to near
        the bottom of frame. Manual calibration has no ankle point, so
        this reuses the same 0.98*height floor lower_body_box() uses as an
        approximation - good enough since a dress's own hem in the garment
        photo determines how far down it actually gets painted anyway.
        """
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0]]
        y0 = min(self.left_shoulder[1], self.right_shoulder[1])
        y1 = image_height * 0.98
        return min(xs), y0, max(xs), y1

    def to_pixels(self, image_width: int, image_height: int) -> "BodyKeypoints":
        """
        Rescale keypoints stored as fractions of the calibration photo's
        width/height (0.0-1.0) into absolute pixel coordinates for a given
        (possibly different-resolution) image of the SAME mannequin/crop.
        This is what makes keypoints calibrated once reusable at inference
        even when the mannequin photo you feed to generate.py isn't the
        exact same pixel dimensions as the one you clicked on in Cell 4.
        """
        def scale(pt):
            if pt is None:
                return None
            return (pt[0] * image_width, pt[1] * image_height)

        return BodyKeypoints(
            left_shoulder=scale(self.left_shoulder),
            right_shoulder=scale(self.right_shoulder),
            left_hip=scale(self.left_hip),
            right_hip=scale(self.right_hip),
            left_knee=scale(self.left_knee),
            right_knee=scale(self.right_knee),
            neck=scale(self.neck),
        )


# ---------- MANUAL calibration (recommended default) ----------

def save_manual_keypoints(mannequin_id: str, keypoints: BodyKeypoints, store_dir: str = "models/mannequin_keypoints"):
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"{mannequin_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(keypoints), f, indent=2)
    return path


def load_manual_keypoints(mannequin_id: str, store_dir: str = "models/mannequin_keypoints") -> BodyKeypoints:
    path = os.path.join(store_dir, f"{mannequin_id}.json")
    with open(path) as f:
        data = json.load(f)
    return BodyKeypoints(**data)


# ---------- AUTO detection (MediaPipe PoseLandmarker) ----------
# Requires the .task model file, downloaded once:
#   wget -O models/pose_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
# (Run this line in Colab - not available in this sandboxed dev environment.)

_POSE_LANDMARK_INDEX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
}


def detect_keypoints_auto(image_np: np.ndarray, model_path: str = "models/pose_landmarker.task") -> BodyKeypoints:
    """
    Auto-detect keypoints using MediaPipe PoseLandmarker.
    image_np: HxWx3 RGB numpy array.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise RuntimeError("No pose detected on mannequin image - fall back to MANUAL calibration for this photo.")

    lm = result.pose_landmarks[0]
    h, w = image_np.shape[:2]

    def pt(idx):
        return (lm[idx].x * w, lm[idx].y * h)

    return BodyKeypoints(
        left_shoulder=pt(_POSE_LANDMARK_INDEX["left_shoulder"]),
        right_shoulder=pt(_POSE_LANDMARK_INDEX["right_shoulder"]),
        left_hip=pt(_POSE_LANDMARK_INDEX["left_hip"]),
        right_hip=pt(_POSE_LANDMARK_INDEX["right_hip"]),
        left_knee=pt(_POSE_LANDMARK_INDEX["left_knee"]),
        right_knee=pt(_POSE_LANDMARK_INDEX["right_knee"]),
    )
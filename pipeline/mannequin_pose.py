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

    def torso_box(self) -> tuple[int, int, int, int]:
        """Bounding box for the torso region (for top garment placement).

        NOTE: this box is bounded strictly by the shoulder/hip points, so it
        does NOT cover the arms at all. Use torso_and_arms_box() instead when
        the garment has sleeves (short or long), or the inpainting mask will
        never include the arm area and the model will render a sleeveless
        top regardless of the source garment photo.
        """
        xs = [self.left_shoulder[0], self.right_shoulder[0], self.left_hip[0], self.right_hip[0]]
        ys = [self.left_shoulder[1], self.right_shoulder[1], self.left_hip[1], self.right_hip[1]]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def torso_and_arms_box(self, sleeve_length: str = "short") -> tuple[int, int, int, int]:
        """Bounding box for torso PLUS the arms, so sleeved garments have
        room to be painted. We don't have elbow/wrist keypoints from manual
        calibration (only shoulder/hip/knee), so the arm extent is estimated
        from shoulder width and torso height:

          - "none"  : same as torso_box() (tank top / sleeveless garment)
          - "short" : widen box sideways to cover the upper arm/bicep
                      and extend it down a bit past the shoulder line
          - "long"  : widen further and extend down closer to where the
                      wrist would be (roughly hip level)

        This is an approximation, not a real arm mask - it's a rectangle
        wide/tall enough that the diffusion inpainting has the pixels it
        needs to draw a sleeve. Good enough for most mannequin poses with
        arms held at/near the sides.
        """
        x0, y0, x1, y1 = self.torso_box()
        shoulder_width = abs(self.right_shoulder[0] - self.left_shoulder[0])
        torso_height = y1 - y0

        if sleeve_length == "none":
            return x0, y0, x1, y1

        if sleeve_length == "short":
            side_pad = shoulder_width * 0.45          # room for upper arm/bicep
            bottom_y = y0 + torso_height * 0.75        # ends a bit above the elbow
        else:  # "long"
            side_pad = shoulder_width * 0.65
            bottom_y = y1 + torso_height * 0.5          # reaches down toward the wrist

        return (
            int(x0 - side_pad),
            int(y0),
            int(x1 + side_pad),
            int(bottom_y),
        )

    def lower_body_box(self, image_height: int) -> tuple[int, int, int, int]:
        """Bounding box for hips-to-ankle region (for bottom garment placement)."""
        xs = [self.left_hip[0], self.right_hip[0]]
        ys_top = min(self.left_hip[1], self.right_hip[1])
        x0, x1 = min(xs) - 20, max(xs) + 20
        y1 = image_height * 0.98
        return int(x0), int(ys_top), int(x1), int(y1)


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

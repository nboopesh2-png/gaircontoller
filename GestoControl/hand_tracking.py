"""
GestoControl - Hand Tracking Module
Encapsulates hand detection and 21 landmark extraction with multi-tier compatibility:
  Tier 1: mediapipe.python.solutions.hands (direct subpackage import for mediapipe 0.10+)
  Tier 2: mediapipe.solutions.hands (legacy import)
  Tier 3: mediapipe.tasks.python.vision.HandLandmarker (modern Tasks API with auto-download)
  Tier 4: OpenCV Skin-Color & Convexity Defects Hand Tracker (graceful fallback)
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# ==============================================================================
# Multi-Tier MediaPipe Importer
# ==============================================================================
MP_SOLUTIONS_HANDS = None
MP_DRAWING_UTILS = None
MP_TASKS_VISION = None
MP_TASKS_PYTHON = None
MP_MODULE = None

# Attempt 1: Direct subpackage import (resolves AttributeError on mediapipe >= 0.10)
try:
    import mediapipe.python.solutions.hands as _mp_hands
    import mediapipe.python.solutions.drawing_utils as _mp_drawing
    MP_SOLUTIONS_HANDS = _mp_hands
    MP_DRAWING_UTILS = _mp_drawing
except Exception:
    pass

# Attempt 2: Top-level mediapipe.solutions import
if MP_SOLUTIONS_HANDS is None:
    try:
        import mediapipe as mp
        MP_MODULE = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            MP_SOLUTIONS_HANDS = mp.solutions.hands
            MP_DRAWING_UTILS = mp.solutions.drawing_utils
    except Exception:
        pass

# Attempt 3: from mediapipe import solutions
if MP_SOLUTIONS_HANDS is None:
    try:
        from mediapipe import solutions
        if hasattr(solutions, "hands"):
            MP_SOLUTIONS_HANDS = solutions.hands
            MP_DRAWING_UTILS = solutions.drawing_utils
    except Exception:
        pass

# Attempt 4: MediaPipe Tasks API
if MP_SOLUTIONS_HANDS is None:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _tasks_py
        from mediapipe.tasks.python import vision as _tasks_vision
        MP_MODULE = mp
        MP_TASKS_PYTHON = _tasks_py
        MP_TASKS_VISION = _tasks_vision
    except Exception:
        pass


# Connections between the 21 MediaPipe hand landmarks
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]

# Landmark indices for fingertips
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class HandData:
    """Encapsulates detected hand data for a single frame."""

    def __init__(
        self,
        landmarks_norm: np.ndarray,         # 21x3 float array [0..1]
        landmarks_px: np.ndarray,           # 21x2 int array (pixel coords)
        palm_center: Tuple[float, float],   # (norm_x, norm_y)
        palm_center_px: Tuple[int, int],    # (px_x, px_y)
        bbox_px: Tuple[int, int, int, int], # (xmin, ymin, xmax, ymax)
        handedness: str = "Right",
        score: float = 0.95
    ):
        self.landmarks_norm = landmarks_norm
        self.landmarks_px = landmarks_px
        self.palm_center = palm_center
        self.palm_center_px = palm_center_px
        self.bbox_px = bbox_px
        self.handedness = handedness
        self.score = score

    @property
    def thumb_tip(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[4, 0]), float(self.landmarks_norm[4, 1]))

    @property
    def index_tip(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[8, 0]), float(self.landmarks_norm[8, 1]))

    @property
    def middle_tip(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[12, 0]), float(self.landmarks_norm[12, 1]))

    @property
    def ring_tip(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[16, 0]), float(self.landmarks_norm[16, 1]))

    @property
    def pinky_tip(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[20, 0]), float(self.landmarks_norm[20, 1]))

    @property
    def wrist(self) -> Tuple[float, float]:
        return (float(self.landmarks_norm[0, 0]), float(self.landmarks_norm[0, 1]))


class HandTracker:
    """
    Manages hand detection across MediaPipe Solutions, MediaPipe Tasks,
    or pure OpenCV skin/contour fallback.
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.70,
        min_tracking_confidence: float = 0.70
    ):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.backend = "NONE"
        self.solutions_hands = None
        self.task_landmarker = None

        # 1. Initialize MediaPipe Solutions (Tiers 1 & 2)
        if MP_SOLUTIONS_HANDS is not None:
            try:
                self.solutions_hands = MP_SOLUTIONS_HANDS.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.backend = "MEDIAPIPE_SOLUTIONS"
                print("[HandTracker] Initialized with MediaPipe Solutions.")
                return
            except Exception as err:
                print(f"[HandTracker] Could not create Solutions.Hands: {err}")

        # 2. Initialize MediaPipe Tasks API (Tier 3)
        if MP_TASKS_VISION is not None and MP_TASKS_PYTHON is not None:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                task_path = os.path.join(base_dir, "hand_landmarker.task")

                # Download model if not yet cached
                if not os.path.exists(task_path):
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    print(f"[HandTracker] Downloading hand_landmarker.task to {task_path}...")
                    import urllib.request
                    urllib.request.urlretrieve(url, task_path)

                if os.path.exists(task_path):
                    base_options = MP_TASKS_PYTHON.BaseOptions(model_asset_path=task_path)
                    options = MP_TASKS_VISION.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=self.max_hands,
                        min_hand_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence
                    )
                    self.task_landmarker = MP_TASKS_VISION.HandLandmarker.create_from_options(options)
                    self.backend = "MEDIAPIPE_TASKS"
                    print("[HandTracker] Initialized with MediaPipe Tasks Landmarker.")
                    return
            except Exception as err:
                print(f"[HandTracker] Could not initialize Tasks HandLandmarker: {err}")

        # 3. Fallback: OpenCV Skin-Color & Convexity Defects Tracker (Tier 4)
        self.backend = "OPENCV_FALLBACK"
        print("[HandTracker] MediaPipe solution unavailable. Operating with OpenCV Computer Vision fallback.")

    def process(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """Processes frame and extracts HandData."""
        if frame_bgr is None:
            return None

        if self.backend == "MEDIAPIPE_SOLUTIONS" and self.solutions_hands is not None:
            return self._process_solutions(frame_bgr)
        elif self.backend == "MEDIAPIPE_TASKS" and self.task_landmarker is not None:
            return self._process_tasks(frame_bgr)
        else:
            return self._process_opencv(frame_bgr)

    def _process_solutions(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """Processes frame via MediaPipe Solutions."""
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        try:
            results = self.solutions_hands.process(rgb_frame)
        except Exception as err:
            print(f"[HandTracker] Solutions error: {err}")
            return None

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = "Right"
        score = 0.95

        if results.multi_handedness and len(results.multi_handedness) > 0:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            score = classification.score

        return self._format_hand_data(hand_landmarks.landmark, w, h, handedness, score)

    def _process_tasks(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """Processes frame via MediaPipe Tasks HandLandmarker."""
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.task_landmarker.detect(mp_image)
            if not results.hand_landmarks:
                return None

            landmarks = results.hand_landmarks[0]
            handedness = "Right"
            score = 0.95
            if results.handedness and len(results.handedness) > 0:
                handedness = results.handedness[0][0].category_name
                score = results.handedness[0][0].score

            return self._format_hand_data(landmarks, w, h, handedness, score)
        except Exception as err:
            print(f"[HandTracker] Tasks error: {err}")
            return None

    def _process_opencv(self, frame_bgr: np.ndarray) -> Optional[HandData]:
        """
        Pure OpenCV fallback tracker using YCrCb skin segmentation,
        contour hull analysis, and geometric landmark reconstruction.
        Ensures the application never crashes even without MediaPipe.
        """
        h, w, _ = frame_bgr.shape

        # Convert to YCrCb space for robust skin color thresholding
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Morphological denoising
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        if area < 6000:  # Minimum hand size
            return None

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(max_contour)

        # Palm center from image moments
        moments = cv2.moments(max_contour)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Convex hull and defect extraction for fingertips
        hull = cv2.convexHull(max_contour, returnPoints=False)
        fingertips = []
        if hull is not None and len(hull) > 3:
            try:
                defects = cv2.convexityDefects(max_contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        # Keep points in upper portion of bounding box
                        if start[1] < cy and d > 3000:
                            fingertips.append(start)
            except Exception:
                pass

        # Sort fingertips by X coordinate (Thumb to Pinky)
        fingertips = sorted(fingertips, key=lambda p: p[0])

        # Synthesize 21 landmarks based on hand bounding geometry
        norm_pts = np.zeros((21, 3), dtype=np.float32)
        px_pts = np.zeros((21, 2), dtype=np.int32)

        # Wrist (0)
        wrist_px = (cx, min(h - 1, y + bh))
        px_pts[0] = wrist_px

        # Assign fingertips or interpolate if not enough detected
        tip_coords = []
        if len(fingertips) >= 5:
            tip_coords = fingertips[:5]
        else:
            # Generate estimated fan of 5 fingers
            base_y = max(0, y + 10)
            xs = np.linspace(x + 10, x + bw - 10, 5)
            tip_coords = [(int(fx), int(base_y)) for fx in xs]

        # Populate fingers: Thumb (1-4), Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
        for f_idx, tip in enumerate(tip_coords):
            base_idx = 1 + f_idx * 4
            tip_idx = base_idx + 3
            # MCP, PIP, DIP, TIP interpolation between palm center and tip
            for step in range(4):
                ratio = (step + 1) / 4.0
                ix = int(cx + (tip[0] - cx) * ratio)
                iy = int(cy + (tip[1] - cy) * ratio)
                cur_idx = base_idx + step
                px_pts[cur_idx] = [ix, iy]

        # Normalized coordinates
        for i in range(21):
            norm_pts[i] = [px_pts[i, 0] / float(w), px_pts[i, 1] / float(h), 0.0]

        palm_center_norm = (cx / float(w), cy / float(h))
        palm_center_px = (cx, cy)
        bbox_px = (x, y, x + bw, y + bh)

        return HandData(
            landmarks_norm=norm_pts,
            landmarks_px=px_pts,
            palm_center=palm_center_norm,
            palm_center_px=palm_center_px,
            bbox_px=bbox_px,
            handedness="Right",
            score=0.85
        )

    def _format_hand_data(self, landmarks, w: int, h: int, handedness: str, score: float) -> HandData:
        """Formats 21 MediaPipe landmark points into structured HandData."""
        norm_pts = np.zeros((21, 3), dtype=np.float32)
        px_pts = np.zeros((21, 2), dtype=np.int32)

        for i, lm in enumerate(landmarks):
            norm_pts[i] = [lm.x, lm.y, getattr(lm, "z", 0.0)]
            px_x = int(np.clip(lm.x * w, 0, w - 1))
            px_y = int(np.clip(lm.y * h, 0, h - 1))
            px_pts[i] = [px_x, px_y]

        palm_indices = [0, 5, 9, 13, 17]
        palm_center_norm = tuple(np.mean(norm_pts[palm_indices, :2], axis=0))
        palm_center_px = (int(np.mean(px_pts[palm_indices, 0])), int(np.mean(px_pts[palm_indices, 1])))

        x_min = int(np.min(px_pts[:, 0]))
        y_min = int(np.min(px_pts[:, 1]))
        x_max = int(np.max(px_pts[:, 0]))
        y_max = int(np.max(px_pts[:, 1]))

        padding = 15
        bbox_px = (
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(w - 1, x_max + padding),
            min(h - 1, y_max + padding)
        )

        return HandData(
            landmarks_norm=norm_pts,
            landmarks_px=px_pts,
            palm_center=palm_center_norm,
            palm_center_px=palm_center_px,
            bbox_px=bbox_px,
            handedness=handedness,
            score=score
        )

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        hand_data: HandData,
        highlight_fingertips: bool = True
    ) -> np.ndarray:
        """Draws custom hand skeleton and landmark markers on the provided frame."""
        if hand_data is None or frame_bgr is None:
            return frame_bgr

        # 1. Connection lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = tuple(hand_data.landmarks_px[start_idx])
            pt2 = tuple(hand_data.landmarks_px[end_idx])
            cv2.line(frame_bgr, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)

        # 2. Joint points
        for i, pt in enumerate(hand_data.landmarks_px):
            pt_t = tuple(pt)
            if i in FINGERTIP_INDICES and highlight_fingertips:
                cv2.circle(frame_bgr, pt_t, 8, (0, 255, 127), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt_t, 10, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, pt_t, 5, (255, 191, 0), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt_t, 6, (40, 40, 40), 1, cv2.LINE_AA)

        # 3. Palm center
        cv2.circle(frame_bgr, hand_data.palm_center_px, 6, (255, 0, 128), -1, cv2.LINE_AA)

        # 4. Bounding box accents
        x1, y1, x2, y2 = hand_data.bbox_px
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (70, 70, 70), 1, cv2.LINE_AA)
        line_len = 16
        color_accent = (0, 215, 255)
        thick = 2
        cv2.line(frame_bgr, (x1, y1), (x1 + line_len, y1), color_accent, thick)
        cv2.line(frame_bgr, (x1, y1), (x1, y1 + line_len), color_accent, thick)
        cv2.line(frame_bgr, (x2, y1), (x2 - line_len, y1), color_accent, thick)
        cv2.line(frame_bgr, (x2, y1), (x2, y1 + line_len), color_accent, thick)
        cv2.line(frame_bgr, (x1, y2), (x1 + line_len, y2), color_accent, thick)
        cv2.line(frame_bgr, (x1, y2), (x1, y2 - line_len), color_accent, thick)
        cv2.line(frame_bgr, (x2, y2), (x2 - line_len, y2), color_accent, thick)
        cv2.line(frame_bgr, (x2, y2), (x2, y2 - line_len), color_accent, thick)

        return frame_bgr

    def close(self):
        """Releases underlying resources."""
        if self.solutions_hands is not None:
            try:
                self.solutions_hands.close()
            except Exception:
                pass
            self.solutions_hands = None
        if self.task_landmarker is not None:
            try:
                self.task_landmarker.close()
            except Exception:
                pass
            self.task_landmarker = None

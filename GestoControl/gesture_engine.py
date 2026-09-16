"""
GestoControl - Gesture Engine Module
Comprehensive gesture recognition engine providing distance calculation,
pinch detection, open palm, fist, swipe detection, finger pointing,
and a gesture-priority resolution system.
"""

from collections import deque
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from hand_tracking import HandData


# Gesture Names
GESTURE_NONE = "NONE"
GESTURE_PINCH = "PINCH"
GESTURE_TWO_FINGER = "TWO FINGER"
GESTURE_OPEN_PALM = "OPEN PALM"
GESTURE_FIST = "FIST"
GESTURE_POINT = "POINT"
GESTURE_SWIPE_UP = "SWIPE UP"
GESTURE_SWIPE_DOWN = "SWIPE DOWN"
GESTURE_SWIPE_LEFT = "SWIPE LEFT"
GESTURE_SWIPE_RIGHT = "SWIPE RIGHT"


def calculate_distance(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    """
    Computes Euclidean distance between two points using NumPy.
    Supports 2D and 3D points.
    """
    arr1 = np.array(point1[:2], dtype=np.float32)
    arr2 = np.array(point2[:2], dtype=np.float32)
    return float(np.linalg.norm(arr1 - arr2))


def detect_hand(landmarks: Optional[np.ndarray]) -> bool:
    """
    Verifies that hand landmarks are properly detected and structured.
    """
    if landmarks is None:
        return False
    if not isinstance(landmarks, np.ndarray):
        return False
    if landmarks.shape[0] < 21:
        return False
    return True


def detect_pinch(landmarks: np.ndarray, threshold: float = 0.055) -> Tuple[bool, float]:
    """
    Detects if the thumb tip (landmark 4) and index finger tip (landmark 8)
    are pinched together within the threshold distance.
    Returns: (is_pinched, distance)
    """
    if not detect_hand(landmarks):
        return False, 1.0

    p_thumb = landmarks[4, :2]
    p_index = landmarks[8, :2]
    dist = calculate_distance(p_thumb, p_index)
    return (dist < threshold), dist


def detect_two_finger(landmarks: np.ndarray, threshold: float = 0.065) -> bool:
    """
    Detects if index finger and middle finger tips are extended together
    while ring and pinky fingers are curled inward.
    Used for Right-Click in Air Mouse mode.
    """
    if not detect_hand(landmarks):
        return False

    # Check that index (8) and middle (12) tips are close
    dist_tips = calculate_distance(landmarks[8, :2], landmarks[12, :2])
    if dist_tips > threshold:
        return False

    wrist = landmarks[0, :2]
    # Check index and middle are extended further from wrist than their PIP joints
    idx_extended = calculate_distance(landmarks[8, :2], wrist) > calculate_distance(landmarks[6, :2], wrist)
    mid_extended = calculate_distance(landmarks[12, :2], wrist) > calculate_distance(landmarks[10, :2], wrist)

    # Check ring and pinky are folded
    ring_folded = calculate_distance(landmarks[16, :2], wrist) < calculate_distance(landmarks[14, :2], wrist) * 1.1
    pinky_folded = calculate_distance(landmarks[20, :2], wrist) < calculate_distance(landmarks[18, :2], wrist) * 1.1

    return bool(idx_extended and mid_extended and ring_folded and pinky_folded)


def detect_open_palm(landmarks: np.ndarray) -> bool:
    """
    Detects an open palm gesture where all 5 fingers are fully extended outward.
    """
    if not detect_hand(landmarks):
        return False

    wrist = landmarks[0, :2]

    # Check thumb extended away from palm
    thumb_dist_wrist = calculate_distance(landmarks[4, :2], wrist)
    thumb_ip_wrist = calculate_distance(landmarks[3, :2], wrist)
    thumb_extended = thumb_dist_wrist > thumb_ip_wrist

    # Check 4 fingers: tip must be further from wrist than PIP joint
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    extended_count = 0
    if thumb_extended:
        extended_count += 1

    for tip, pip in zip(finger_tips, finger_pips):
        tip_dist = calculate_distance(landmarks[tip, :2], wrist)
        pip_dist = calculate_distance(landmarks[pip, :2], wrist)
        if tip_dist > pip_dist * 1.08:
            extended_count += 1

    return extended_count >= 5


def detect_fist(landmarks: np.ndarray) -> bool:
    """
    Detects a closed fist gesture where all 4 fingers are curled down towards the palm.
    """
    if not detect_hand(landmarks):
        return False

    wrist = landmarks[0, :2]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    curled_count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        tip_dist = calculate_distance(landmarks[tip, :2], wrist)
        pip_dist = calculate_distance(landmarks[pip, :2], wrist)
        # In a fist, the tip is curled closer to wrist than or equal to PIP joint
        if tip_dist < pip_dist * 1.05:
            curled_count += 1

    # Thumb tip close to index MCP (5) or curled
    thumb_to_index_mcp = calculate_distance(landmarks[4, :2], landmarks[5, :2])
    thumb_curled = thumb_to_index_mcp < 0.12

    return (curled_count >= 4) and thumb_curled


def detect_point(landmarks: np.ndarray) -> bool:
    """
    Detects a pointing gesture where only the index finger is extended,
    and middle, ring, and pinky fingers are curled.
    """
    if not detect_hand(landmarks):
        return False

    wrist = landmarks[0, :2]

    # Index finger must be extended
    idx_tip_dist = calculate_distance(landmarks[8, :2], wrist)
    idx_pip_dist = calculate_distance(landmarks[6, :2], wrist)
    idx_extended = idx_tip_dist > idx_pip_dist * 1.15

    # Middle, Ring, Pinky must be curled
    other_tips = [12, 16, 20]
    other_pips = [10, 14, 18]
    curled = 0
    for tip, pip in zip(other_tips, other_pips):
        if calculate_distance(landmarks[tip, :2], wrist) < calculate_distance(landmarks[pip, :2], wrist) * 1.05:
            curled += 1

    return bool(idx_extended and curled >= 3)


def detect_finger_direction(
    point_start: Tuple[float, float],
    point_end: Tuple[float, float]
) -> Tuple[str, float]:
    """
    Calculates movement direction vector and angle between two points.
    Returns: (direction_label, angle_degrees)
    """
    dx = point_end[0] - point_start[0]
    dy = point_end[1] - point_start[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))

    if abs(dx) > abs(dy):
        direction = "RIGHT" if dx > 0 else "LEFT"
    else:
        direction = "DOWN" if dy > 0 else "UP"

    return direction, angle


class MotionHistory:
    """
    Tracks hand trajectory across multiple frames to reliably detect swipes
    without single-frame false triggers.
    """

    def __init__(self, max_points: int = 15, max_time_window: float = 0.5):
        self.max_points = max_points
        self.max_time_window = max_time_window
        self.history: Deque[Tuple[float, Tuple[float, float]]] = deque(maxlen=max_points)

    def add(self, point: Tuple[float, float]):
        """Adds current normalized (x, y) palm center with timestamp."""
        now = time.time()
        self.history.append((now, point))
        self._prune(now)

    def _prune(self, current_time: float):
        """Removes entries older than max_time_window."""
        while self.history and (current_time - self.history[0][0] > self.max_time_window):
            self.history.popleft()

    def clear(self):
        """Clears trajectory history (e.g. after a swipe is fired)."""
        self.history.clear()

    @property
    def count(self) -> int:
        return len(self.history)


def detect_swipe(
    motion_history: MotionHistory,
    min_distance: float = 0.10,
    min_frames: int = 5
) -> Optional[str]:
    """
    Detects hand movement across multiple frames.
    Analyzes net displacement, velocity, and trajectory linearity.
    Returns: 'SWIPE UP', 'SWIPE DOWN', 'SWIPE LEFT', 'SWIPE RIGHT', or None.
    """
    if motion_history.count < min_frames:
        return None

    # Compare oldest point in window with the most recent point
    t_start, pt_start = motion_history.history[0]
    t_end, pt_end = motion_history.history[-1]

    dt = t_end - t_start
    if dt <= 0.05:
        return None

    dx = pt_end[0] - pt_start[0]
    dy = pt_end[1] - pt_start[1]
    dist = float(np.sqrt(dx ** 2 + dy ** 2))

    if dist < min_distance:
        return None

    # Calculate speed
    speed = dist / dt
    if speed < 0.20:
        return None  # Too slow, just slow hand repositioning

    # Check dominance of axis to ensure intentional linear swipe
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # In screen coordinates: Y=0 is top, Y=1 is bottom.
    # Hand moving UP -> dy is negative. Hand moving DOWN -> dy is positive.
    if abs_dy > 1.25 * abs_dx:
        if dy < 0:
            return GESTURE_SWIPE_UP
        else:
            return GESTURE_SWIPE_DOWN
    elif abs_dx > 1.25 * abs_dy:
        if dx < 0:
            return GESTURE_SWIPE_LEFT
        else:
            return GESTURE_SWIPE_RIGHT

    return None


class GestureEngine:
    """
    Dedicated gesture recognition engine enforcing Gesture Priority
    and debounce/cooldown timing.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.055,
        two_finger_threshold: float = 0.065,
        swipe_min_distance: float = 0.10,
        swipe_cooldown: float = 0.65,
        action_cooldown: float = 0.45
    ):
        self.pinch_threshold = pinch_threshold
        self.two_finger_threshold = two_finger_threshold
        self.swipe_min_distance = swipe_min_distance
        self.swipe_cooldown = swipe_cooldown
        self.action_cooldown = action_cooldown

        self.motion_history = MotionHistory(max_points=16, max_time_window=0.45)
        self.last_swipe_time = 0.0
        self.last_action_time = 0.0
        self.last_gesture = GESTURE_NONE
        self.current_confidence = 0.0

    def update_settings(
        self,
        pinch_threshold: Optional[float] = None,
        swipe_min_distance: Optional[float] = None,
        swipe_cooldown: Optional[float] = None,
        action_cooldown: Optional[float] = None
    ):
        """Updates runtime recognition thresholds."""
        if pinch_threshold is not None:
            self.pinch_threshold = pinch_threshold
        if swipe_min_distance is not None:
            self.swipe_min_distance = swipe_min_distance
        if swipe_cooldown is not None:
            self.swipe_cooldown = swipe_cooldown
        if action_cooldown is not None:
            self.action_cooldown = action_cooldown

    def detect_gesture(
        self,
        hand_data: Optional[HandData],
        current_mode: str = "REELS"
    ) -> Tuple[str, Dict[str, any]]:
        """
        Processes the current hand data through the gesture priority pipeline.
        Priority Hierarchy:
          1. PINCH (highest priority - prevents accidental swipes while pinching)
          2. TWO FINGER (for right-click in Air Mouse)
          3. FIST (distinct closed state)
          4. OPEN PALM (all fingers open)
          5. SWIPE (requires movement across multiple frames & cooldown)
          6. POINT (single index extended)
          7. NONE

        Returns: (gesture_name, metadata_dict)
        """
        now = time.time()
        metadata = {
            "hand_detected": False,
            "pinch_distance": 1.0,
            "confidence": 0.0,
            "can_trigger": False
        }

        if hand_data is None:
            self.motion_history.clear()
            self.last_gesture = GESTURE_NONE
            return GESTURE_NONE, metadata

        metadata["hand_detected"] = True
        landmarks = hand_data.landmarks_norm
        palm_center = hand_data.palm_center

        # Record palm trajectory
        self.motion_history.add(palm_center)

        # 1. PRIORITY 1: PINCH
        is_pinch, pinch_dist = detect_pinch(landmarks, self.pinch_threshold)
        metadata["pinch_distance"] = pinch_dist

        if is_pinch:
            # Clear motion history so pinch movement isn't mistaken for swipe
            self.motion_history.clear()
            self.last_gesture = GESTURE_PINCH
            metadata["confidence"] = max(0.0, min(1.0, 1.0 - (pinch_dist / self.pinch_threshold)))
            metadata["can_trigger"] = (now - self.last_action_time) > self.action_cooldown
            return GESTURE_PINCH, metadata

        # 2. PRIORITY 2: TWO FINGER (especially relevant in Air Mouse)
        if current_mode == "AIR MOUSE" and detect_two_finger(landmarks, self.two_finger_threshold):
            self.motion_history.clear()
            self.last_gesture = GESTURE_TWO_FINGER
            metadata["confidence"] = 0.90
            metadata["can_trigger"] = (now - self.last_action_time) > self.action_cooldown
            return GESTURE_TWO_FINGER, metadata

        # 3. PRIORITY 3: FIST
        if detect_fist(landmarks):
            self.motion_history.clear()
            self.last_gesture = GESTURE_FIST
            metadata["confidence"] = 0.92
            metadata["can_trigger"] = (now - self.last_action_time) > self.action_cooldown
            return GESTURE_FIST, metadata

        # 4. PRIORITY 4: OPEN PALM
        if detect_open_palm(landmarks):
            # If the hand is an open palm and moving fast, let's check if it's a swipe
            # Otherwise it's an OPEN PALM
            swipe = detect_swipe(self.motion_history, self.swipe_min_distance)
            if swipe is not None and (now - self.last_swipe_time) > self.swipe_cooldown:
                self.motion_history.clear()
                self.last_swipe_time = now
                self.last_gesture = swipe
                metadata["confidence"] = 0.88
                metadata["can_trigger"] = True
                return swipe, metadata

            self.last_gesture = GESTURE_OPEN_PALM
            metadata["confidence"] = 0.94
            metadata["can_trigger"] = (now - self.last_action_time) > self.action_cooldown
            return GESTURE_OPEN_PALM, metadata

        # 5. PRIORITY 5: SWIPE DETECTION
        swipe = detect_swipe(self.motion_history, self.swipe_min_distance)
        if swipe is not None:
            if (now - self.last_swipe_time) > self.swipe_cooldown:
                self.motion_history.clear()
                self.last_swipe_time = now
                self.last_gesture = swipe
                metadata["confidence"] = 0.85
                metadata["can_trigger"] = True
                return swipe, metadata
            else:
                # In cooldown
                return GESTURE_NONE, metadata

        # 6. PRIORITY 6: POINT (Index finger only)
        if detect_point(landmarks):
            self.last_gesture = GESTURE_POINT
            metadata["confidence"] = 0.89
            metadata["can_trigger"] = True
            return GESTURE_POINT, metadata

        self.last_gesture = GESTURE_NONE
        return GESTURE_NONE, metadata

    def record_action_executed(self):
        """Notifies the engine that an action was triggered to enforce cooldown."""
        self.last_action_time = time.time()

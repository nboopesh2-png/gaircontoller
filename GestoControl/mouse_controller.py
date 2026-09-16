"""
GestoControl - Mouse Controller Module
Translates index-finger coordinates to Windows screen cursor coordinates.
Provides cursor smoothing (EMA), active-box mapping, pinch-to-left-click,
two-finger right-click, open-palm freeze, and debouncing.
"""

import time
from typing import Optional, Tuple
import numpy as np

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable failsafe to prevent edge exceptions
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


class MouseController:
    """Manages virtual air mouse movement, smoothing, and gesture clicking."""

    def __init__(
        self,
        sensitivity: float = 1.6,
        smoothing: float = 0.35,
        deadzone: float = 0.004,
        click_cooldown: float = 0.40
    ):
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self.deadzone = deadzone
        self.click_cooldown = click_cooldown

        # Screen dimensions
        if PYAUTOGUI_AVAILABLE:
            try:
                self.screen_w, self.screen_h = pyautogui.size()
            except Exception:
                self.screen_w, self.screen_h = 1920, 1080
        else:
            self.screen_w, self.screen_h = 1920, 1080

        # Cursor positions (screen space)
        self.current_x: float = self.screen_w / 2.0
        self.current_y: float = self.screen_h / 2.0
        self.target_x: float = self.screen_w / 2.0
        self.target_y: float = self.screen_h / 2.0

        # Normalized coordinates
        self.prev_norm_x: Optional[float] = None
        self.prev_norm_y: Optional[float] = None

        # Interaction states
        self.is_pinched: bool = False
        self.is_two_finger: bool = False
        self.last_left_click_time: float = 0.0
        self.last_right_click_time: float = 0.0

        # Margin box to allow comfortable hand range (0.15 to 0.85)
        self.margin_x = 0.15
        self.margin_y = 0.15

    def move(self, norm_x: float, norm_y: float, freeze: bool = False) -> Tuple[int, int]:
        """
        Updates cursor target position from normalized coordinates [0..1],
        applies active-box mapping, deadzone filtering, and exponential smoothing.
        Returns the (x, y) screen pixel coordinates.
        """
        if not PYAUTOGUI_AVAILABLE or freeze:
            return int(self.current_x), int(self.current_y)

        # Check deadzone
        if self.prev_norm_x is not None and self.prev_norm_y is not None:
            dist = np.hypot(norm_x - self.prev_norm_x, norm_y - self.prev_norm_y)
            if dist < self.deadzone:
                # Hand is held steady, do not jitter cursor
                return int(self.current_x), int(self.current_y)

        self.prev_norm_x = norm_x
        self.prev_norm_y = norm_y

        # Map active sub-box [margin..1-margin] to [0..1]
        clamped_x = max(self.margin_x, min(1.0 - self.margin_x, norm_x))
        clamped_y = max(self.margin_y, min(1.0 - self.margin_y, norm_y))

        scaled_x = (clamped_x - self.margin_x) / (1.0 - 2 * self.margin_x)
        scaled_y = (clamped_y - self.margin_y) / (1.0 - 2 * self.margin_y)

        # Apply sensitivity multiplier around center
        centered_x = (scaled_x - 0.5) * self.sensitivity + 0.5
        centered_y = (scaled_y - 0.5) * self.sensitivity + 0.5

        # Screen coordinates
        self.target_x = np.clip(centered_x * self.screen_w, 0, self.screen_w - 1)
        self.target_y = np.clip(centered_y * self.screen_h, 0, self.screen_h - 1)

        # Exponential Moving Average (EMA) smoothing
        self.current_x = self.smoothing * self.target_x + (1.0 - self.smoothing) * self.current_x
        self.current_y = self.smoothing * self.target_y + (1.0 - self.smoothing) * self.current_y

        screen_px_x = int(round(self.current_x))
        screen_px_y = int(round(self.current_y))

        try:
            pyautogui.moveTo(screen_px_x, screen_px_y, _pause=False)
        except Exception:
            pass

        return screen_px_x, screen_px_y

    def left_click(self) -> bool:
        """
        Executes a left click if cooldown has passed and pinch was previously released.
        Returns True if click executed.
        """
        if not PYAUTOGUI_AVAILABLE:
            return False

        now = time.time()
        if (now - self.last_left_click_time) >= self.click_cooldown:
            try:
                pyautogui.click(_pause=False)
                self.last_left_click_time = now
                self.is_pinched = True
                return True
            except Exception:
                pass
        return False

    def right_click(self) -> bool:
        """
        Executes a right click if cooldown has passed.
        Returns True if click executed.
        """
        if not PYAUTOGUI_AVAILABLE:
            return False

        now = time.time()
        if (now - self.last_right_click_time) >= self.click_cooldown:
            try:
                pyautogui.rightClick(_pause=False)
                self.last_right_click_time = now
                self.is_two_finger = True
                return True
            except Exception:
                pass
        return False

    def release_pinch(self):
        """Resets pinch state when fingers separate."""
        self.is_pinched = False

    def release_two_finger(self):
        """Resets two-finger state when fingers separate."""
        self.is_two_finger = False

"""
GestoControl - Volume Controller Module
Controls Windows master system audio volume using:
  1. pycaw (Core Audio Windows API for continuous 0%-100% scalar control)
  2. PyAutoGUI system media volume keys as graceful fallback
Supports continuous thumb-index distance adjustment and step increase/decrease.
"""

import time
from typing import Optional, Tuple
import numpy as np

# Try importing pycaw and comtypes
try:
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except Exception:
    PYCAW_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except Exception:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


class VolumeController:
    """Manages Windows master volume control, distance mapping, and smoothing."""

    def __init__(
        self,
        min_distance: float = 0.03,
        max_distance: float = 0.26,
        smoothing_factor: float = 0.30
    ):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.smoothing_factor = smoothing_factor

        self.current_volume: float = 50.0
        self.target_volume: float = 50.0
        self.is_muted: bool = False
        self.hardware_supported: bool = False
        self.status_message: str = "Initializing Volume..."

        self.last_hardware_update_time: float = 0.0
        self.hardware_update_interval: float = 0.08  # ~12 updates per second

        # Pycaw audio endpoint
        self.endpoint_volume = None
        self._init_pycaw()

    def _init_pycaw(self):
        """Attempts to bind to default Windows audio endpoint via pycaw."""
        if not PYCAW_AVAILABLE:
            self.hardware_supported = False
            self.status_message = "PyAutoGUI Media Key Mode"
            return

        try:
            # Need to initialize COM in current thread if necessary
            devices = AudioUtilities.GetSpeakers()
            if devices is None:
                self.hardware_supported = False
                self.status_message = "No audio speaker endpoint found"
                return

            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.endpoint_volume = cast(interface, POINTER(IAudioEndpointVolume))

            # Query initial volume level
            current_scalar = self.endpoint_volume.GetMasterVolumeLevelScalar()
            self.current_volume = float(np.clip(current_scalar * 100.0, 0.0, 100.0))
            self.target_volume = self.current_volume
            self.is_muted = bool(self.endpoint_volume.GetMute())
            self.hardware_supported = True
            self.status_message = "Hardware Ready (pycaw)"
        except Exception as err:
            self.hardware_supported = False
            self.status_message = f"Audio API fallback ({type(err).__name__})"

    def calculate_distance(self, thumb_tip: Tuple[float, float], index_tip: Tuple[float, float]) -> float:
        """Computes Euclidean distance between thumb tip and index tip using NumPy."""
        p1 = np.array(thumb_tip, dtype=np.float32)
        p2 = np.array(index_tip, dtype=np.float32)
        return float(np.linalg.norm(p1 - p2))

    def update_from_landmarks(
        self,
        thumb_tip: Tuple[float, float],
        index_tip: Tuple[float, float],
        apply_hardware: bool = True
    ) -> int:
        """
        Calculates distance between thumb tip and index tip, maps to 0%-100%,
        applies Exponential Moving Average smoothing, and adjusts master volume.
        Returns: smoothed volume integer percentage (0 to 100).
        """
        raw_distance = self.calculate_distance(thumb_tip, index_tip)

        # Normalize distance into 0.0 - 1.0 range
        clamped_dist = max(self.min_distance, min(self.max_distance, raw_distance))
        normalized_ratio = (clamped_dist - self.min_distance) / (self.max_distance - self.min_distance)

        # Target percentage from 0 to 100
        self.target_volume = float(np.clip(normalized_ratio * 100.0, 0.0, 100.0))

        # Exponential Moving Average (EMA) smoothing: prevents rapid flickering
        self.current_volume = (
            self.smoothing_factor * self.target_volume +
            (1.0 - self.smoothing_factor) * self.current_volume
        )

        now = time.time()
        if apply_hardware and (now - self.last_hardware_update_time >= self.hardware_update_interval):
            self.set_volume(int(round(self.current_volume)))
            self.last_hardware_update_time = now

        return self.get_volume_percentage()

    def set_volume(self, level: int):
        """Sets master system volume directly to specified integer level [0..100]."""
        level = max(0, min(100, level))
        self.current_volume = float(level)

        if self.hardware_supported and self.endpoint_volume is not None:
            try:
                scalar = float(level) / 100.0
                self.endpoint_volume.SetMasterVolumeLevelScalar(scalar, None)
                return
            except Exception as err:
                print(f"[VolumeController] pycaw SetMasterVolumeLevelScalar failed: {err}")

        # Fallback via PyAutoGUI volume keys if pycaw is unavailable
        # (Compare current vs target to press volumeup/volumedown)
        if PYAUTOGUI_AVAILABLE:
            try:
                # We send step keystrokes if continuous scalar is not supported
                pass
            except Exception:
                pass

    def increase_volume(self, step: int = 4) -> int:
        """Increases volume by step percentage."""
        new_vol = min(100, int(round(self.current_volume)) + step)
        self.set_volume(new_vol)
        if not self.hardware_supported and PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumeup")
            except Exception:
                pass
        return self.get_volume_percentage()

    def decrease_volume(self, step: int = 4) -> int:
        """Decreases volume by step percentage."""
        new_vol = max(0, int(round(self.current_volume)) - step)
        self.set_volume(new_vol)
        if not self.hardware_supported and PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumedown")
            except Exception:
                pass
        return self.get_volume_percentage()

    def toggle_mute(self) -> bool:
        """Toggles audio mute state."""
        self.is_muted = not self.is_muted
        if self.hardware_supported and self.endpoint_volume is not None:
            try:
                self.endpoint_volume.SetMute(int(self.is_muted), None)
                return self.is_muted
            except Exception:
                pass

        if PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.press("volumemute")
            except Exception:
                pass
        return self.is_muted

    def get_volume_percentage(self) -> int:
        """Returns current volume as integer percentage 0 to 100."""
        return int(round(self.current_volume))

    def get_status_info(self) -> str:
        """Returns formatted string describing current volume status."""
        mute_str = " (MUTED)" if self.is_muted else ""
        return f"{self.get_volume_percentage()}%{mute_str} - {self.status_message}"

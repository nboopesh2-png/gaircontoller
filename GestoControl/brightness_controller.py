"""
GestoControl - Brightness Controller Module
Controls laptop screen brightness using the distance between Thumb Tip and Index Finger Tip.
Provides Windows hardware integration, exponential smoothing, and graceful fallback.
"""

import subprocess
import time
from typing import Optional, Tuple
import numpy as np

# Try importing screen_brightness_control
try:
    import screen_brightness_control as sbc
    SBC_AVAILABLE = True
except ImportError:
    sbc = None
    SBC_AVAILABLE = False


class BrightnessController:
    """Manages distance-to-brightness mapping, smoothing, and hardware communication."""

    def __init__(
        self,
        min_distance: float = 0.03,
        max_distance: float = 0.26,
        smoothing_factor: float = 0.30
    ):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.smoothing_factor = smoothing_factor

        self.current_brightness: float = 50.0
        self.target_brightness: float = 50.0
        self.hardware_supported: bool = True
        self.hardware_status_message: str = "Hardware Ready"
        self.last_hardware_update_time: float = 0.0
        self.hardware_update_interval: float = 0.15  # Update OS brightness at most ~6 times/sec

        # Probe initial brightness
        self._probe_initial_brightness()

    def _probe_initial_brightness(self):
        """Attempts to read current display brightness from the OS."""
        if SBC_AVAILABLE:
            try:
                val = sbc.get_brightness()
                if isinstance(val, list) and len(val) > 0:
                    self.current_brightness = float(val[0])
                    self.target_brightness = self.current_brightness
                    self.hardware_supported = True
                    self.hardware_status_message = "Connected (screen-brightness-control)"
                    return
                elif isinstance(val, (int, float)):
                    self.current_brightness = float(val)
                    self.target_brightness = self.current_brightness
                    self.hardware_supported = True
                    self.hardware_status_message = "Connected (screen-brightness-control)"
                    return
            except Exception as err:
                # Often occurs on desktop monitors without DDC-CI enabled or VMs
                self.hardware_supported = False
                self.hardware_status_message = f"DDC/CI not supported on this display ({type(err).__name__}). Using Virtual Mode."

        # Fallback probe via WMI
        try:
            cmd = "Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness | Select-Object -ExpandProperty CurrentBrightness"
            output = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=2).strip()
            if output.isdigit():
                self.current_brightness = float(output)
                self.target_brightness = self.current_brightness
                self.hardware_supported = True
                self.hardware_status_message = "Connected (WMI)"
                return
        except Exception:
            pass

        # If we reach here, hardware brightness modification is not directly available
        self.hardware_supported = False
        if not self.hardware_status_message.startswith("DDC/CI"):
            self.hardware_status_message = "Hardware brightness unavailable. Operating in Virtual Mode."

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
    ) -> float:
        """
        Calculates distance, maps to 0%-100%, smooths value, and updates display brightness.
        Returns the smoothed brightness value (0-100).
        """
        raw_distance = self.calculate_distance(thumb_tip, index_tip)

        # Normalize distance into 0.0 - 1.0 range
        clamped_dist = max(self.min_distance, min(self.max_distance, raw_distance))
        normalized_ratio = (clamped_dist - self.min_distance) / (self.max_distance - self.min_distance)

        # Target percentage from 0 to 100
        self.target_brightness = float(np.clip(normalized_ratio * 100.0, 0.0, 100.0))

        # Exponential Moving Average (EMA) smoothing: prevents rapid flickering
        self.current_brightness = (
            self.smoothing_factor * self.target_brightness +
            (1.0 - self.smoothing_factor) * self.current_brightness
        )

        # Apply to hardware if enabled and interval has elapsed
        now = time.time()
        if apply_hardware and (now - self.last_hardware_update_time >= self.hardware_update_interval):
            self._apply_hardware_brightness(int(round(self.current_brightness)))
            self.last_hardware_update_time = now

        return self.get_brightness_percentage()

    def _apply_hardware_brightness(self, level: int):
        """Sends the brightness command to the Windows display."""
        if not self.hardware_supported:
            return

        level = max(0, min(100, level))

        if SBC_AVAILABLE:
            try:
                sbc.set_brightness(level)
                return
            except Exception as err:
                # Mark as unsupported if setting failed
                self.hardware_supported = False
                self.hardware_status_message = f"Cannot set brightness: {err}. Showing Virtual Value."

        # Fallback via PowerShell WMI
        try:
            cmd = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {level})"
            subprocess.run(["powershell", "-NoProfile", "-Command", cmd], timeout=1, capture_output=True)
        except Exception:
            pass

    def get_brightness_percentage(self) -> int:
        """Returns integer percentage 0 to 100."""
        return int(round(self.current_brightness))

    def get_status_info(self) -> str:
        """Returns formatted string describing current brightness status."""
        return f"{self.get_brightness_percentage()}% ({self.hardware_status_message})"

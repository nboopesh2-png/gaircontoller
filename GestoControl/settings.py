"""
GestoControl - Settings Module
Manages application configuration, user preferences, and JSON persistence.
"""

import json
import os
from typing import Any, Dict


DEFAULT_SETTINGS: Dict[str, Any] = {
    # Camera configuration
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "flip_camera": True,

    # Master Control
    "gesture_control_enabled": True,

    # Operational Mode: 'BRIGHTNESS', 'REELS', 'AIR MOUSE', 'AIR KEYBOARD'
    "current_mode": "REELS",

    # Gesture Thresholds
    "pinch_threshold": 0.055,        # Normalized distance between thumb and index tip
    "two_finger_threshold": 0.065,   # Distance between index and middle tips for right-click
    "swipe_min_distance": 0.10,      # Minimum normalized distance traveled to register a swipe
    "swipe_cooldown": 0.65,          # Cooldown in seconds between successive swipes
    "action_cooldown": 0.45,         # Cooldown in seconds between general actions

    # Air Mouse Tuning
    "mouse_sensitivity": 1.6,        # Speed multiplier for mouse cursor
    "mouse_smoothing": 0.35,         # Smoothing weight (EMA alpha: lower is smoother, higher is faster)
    "mouse_deadzone": 0.004,         # Deadzone to prevent micro-jitter

    # Brightness Tuning
    "brightness_min_dist": 0.03,     # Normalized thumb-index distance mapped to 0%
    "brightness_max_dist": 0.26,     # Normalized thumb-index distance mapped to 100%
    "brightness_smoothing": 0.25,

    # Volume Tuning
    "volume_min_dist": 0.03,         # Normalized thumb-index distance mapped to 0%
    "volume_max_dist": 0.26,         # Normalized thumb-index distance mapped to 100%
    "volume_smoothing": 0.30,

    # Air Keyboard Tuning
    "keyboard_debounce": 0.50,       # Delay in seconds to prevent repeated keystrokes
    "keyboard_hover_alpha": 0.35,

    # Key Mappings for Reels / Shorts
    "reels_keys": {
        "NEXT": "down",
        "PREVIOUS": "up",
        "PLAY_PAUSE": "space",
        "MUTE": "m",
        "LIKE": "l"
    },

    # System Media Key Mappings
    "media_keys": {
        "PLAY_PAUSE": "playpause",
        "NEXT": "nexttrack",
        "PREVIOUS": "prevtrack",
        "VOLUME_UP": "volumeup",
        "VOLUME_DOWN": "volumedown"
    }
}


class SettingsManager:
    """Handles loading, saving, and updating runtime configuration."""

    def __init__(self, file_path: str = None):
        if file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.file_path = os.path.join(base_dir, "settings.json")
        else:
            self.file_path = file_path

        self.settings: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """Loads settings from disk, filling in any missing keys with defaults."""
        self.settings = DEFAULT_SETTINGS.copy()
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.settings.update(data)
            except Exception as err:
                print(f"[SettingsManager] Error loading settings from {self.file_path}: {err}. Using defaults.")
        return self.settings

    def save(self) -> bool:
        """Persists current settings dictionary to JSON file."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as err:
            print(f"[SettingsManager] Failed to save settings: {err}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value."""
        return self.settings.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value in memory."""
        self.settings[key] = value

    def reset_to_defaults(self) -> None:
        """Resets all settings back to default values and saves to disk."""
        self.settings = DEFAULT_SETTINGS.copy()
        self.save()

"""
GestoControl - System Media Controller Module
Controls media playback (Spotify, VLC, YouTube, Windows Media Player)
using dedicated OS media keys and PyAutoGUI shortcuts.
"""

import time
from typing import Dict, Optional, Tuple

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


DEFAULT_MEDIA_MAPPING = {
    "OPEN PALM": {
        "action_name": "PLAY / PAUSE",
        "keys": ["playpause"],
        "cooldown": 0.70
    },
    "FIST": {
        "action_name": "PAUSE / STOP",
        "keys": ["playpause"],
        "cooldown": 0.70
    },
    "SWIPE RIGHT": {
        "action_name": "NEXT TRACK",
        "keys": ["nexttrack"],
        "cooldown": 0.65
    },
    "SWIPE LEFT": {
        "action_name": "PREVIOUS TRACK",
        "keys": ["prevtrack"],
        "cooldown": 0.65
    },
    "SWIPE UP": {
        "action_name": "VOLUME UP",
        "keys": ["volumeup", "volumeup"],
        "cooldown": 0.40
    },
    "SWIPE DOWN": {
        "action_name": "VOLUME DOWN",
        "keys": ["volumedown", "volumedown"],
        "cooldown": 0.40
    }
}


class MediaController:
    """Dispatches media keys or shortcut keystrokes based on recognized gestures."""

    def __init__(self, key_bindings: Optional[Dict[str, str]] = None):
        self.mapping = DEFAULT_MEDIA_MAPPING.copy()

        if key_bindings:
            self.update_keybindings(key_bindings)

        self.last_execution_time: Dict[str, float] = {}
        self.last_action_name: str = "Ready"

    def update_keybindings(self, key_bindings: Dict[str, str]):
        """Allows overriding system media keys with custom keyboard shortcuts."""
        if "PLAY_PAUSE" in key_bindings:
            self.mapping["OPEN PALM"]["keys"] = [key_bindings["PLAY_PAUSE"]]
        if "NEXT" in key_bindings:
            self.mapping["SWIPE RIGHT"]["keys"] = [key_bindings["NEXT"]]
        if "PREVIOUS" in key_bindings:
            self.mapping["SWIPE LEFT"]["keys"] = [key_bindings["PREVIOUS"]]
        if "VOLUME_UP" in key_bindings:
            self.mapping["SWIPE UP"]["keys"] = [key_bindings["VOLUME_UP"]]
        if "VOLUME_DOWN" in key_bindings:
            self.mapping["SWIPE DOWN"]["keys"] = [key_bindings["VOLUME_DOWN"]]

    def execute_gesture(self, gesture: str) -> Tuple[bool, str]:
        """
        Executes the media action mapped to the gesture.
        Returns: (was_executed, action_name)
        """
        if gesture not in self.mapping:
            return False, self.last_action_name

        config = self.mapping[gesture]
        action_name = config["action_name"]
        cooldown = config["cooldown"]
        keys = config["keys"]

        now = time.time()
        last_time = self.last_execution_time.get(gesture, 0.0)

        if (now - last_time) < cooldown:
            return False, self.last_action_name

        self.last_execution_time[gesture] = now
        self.last_action_name = action_name

        if PYAUTOGUI_AVAILABLE:
            try:
                for k in keys:
                    pyautogui.press(k)
            except Exception as err:
                print(f"[MediaController] Media key error: {err}")

        return True, action_name

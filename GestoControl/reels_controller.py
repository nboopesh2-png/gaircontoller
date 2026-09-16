"""
GestoControl - Reels & Short Video Controller Module
Controls short-form video platforms (Instagram Reels, YouTube Shorts, TikTok, Reddit)
using configurable keyboard shortcuts and gesture triggers via PyAutoGUI.
"""

import time
from typing import Dict, Optional, Tuple

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


DEFAULT_REELS_MAPPING = {
    "SWIPE UP": {
        "action_name": "NEXT REEL",
        "keys": ["down"],
        "cooldown": 0.65
    },
    "SWIPE DOWN": {
        "action_name": "PREVIOUS REEL",
        "keys": ["up"],
        "cooldown": 0.65
    },
    "SWIPE LEFT": {
        "action_name": "PREVIOUS CONTENT",
        "keys": ["left"],
        "cooldown": 0.60
    },
    "SWIPE RIGHT": {
        "action_name": "NEXT CONTENT",
        "keys": ["right"],
        "cooldown": 0.60
    },
    "OPEN PALM": {
        "action_name": "PLAY / PAUSE",
        "keys": ["space"],
        "cooldown": 0.70
    },
    "FIST": {
        "action_name": "PAUSE / MUTE",
        "keys": ["k"],
        "cooldown": 0.70
    },
    "PINCH": {
        "action_name": "LIKE VIDEO",
        "keys": ["l"],
        "cooldown": 0.80
    }
}


class ReelsController:
    """Dispatches configurable keyboard shortcuts based on recognized hand gestures."""

    def __init__(self, key_bindings: Optional[Dict[str, str]] = None):
        self.mapping = DEFAULT_REELS_MAPPING.copy()

        # Allow overriding keys from settings
        if key_bindings:
            self.update_keybindings(key_bindings)

        self.last_execution_time: Dict[str, float] = {}
        self.last_action_name: str = "Ready"

    def update_keybindings(self, key_bindings: Dict[str, str]):
        """Customizes key commands for different platforms."""
        if "NEXT" in key_bindings:
            self.mapping["SWIPE UP"]["keys"] = [key_bindings["NEXT"]]
        if "PREVIOUS" in key_bindings:
            self.mapping["SWIPE DOWN"]["keys"] = [key_bindings["PREVIOUS"]]
        if "PLAY_PAUSE" in key_bindings:
            self.mapping["OPEN PALM"]["keys"] = [key_bindings["PLAY_PAUSE"]]
        if "MUTE" in key_bindings:
            self.mapping["FIST"]["keys"] = [key_bindings["MUTE"]]
        if "LIKE" in key_bindings:
            self.mapping["PINCH"]["keys"] = [key_bindings["LIKE"]]

    def execute_gesture(self, gesture: str) -> Tuple[bool, str]:
        """
        Executes the action mapped to the given gesture if cooldown has elapsed.
        Returns: (was_executed, action_description)
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

        # Execute shortcut via PyAutoGUI
        if PYAUTOGUI_AVAILABLE:
            try:
                for k in keys:
                    pyautogui.press(k)
            except Exception as err:
                print(f"[ReelsController] Shortcut error: {err}")

        return True, action_name

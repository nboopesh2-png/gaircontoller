"""
GestoControl - Virtual Keyboard Controller Module
Provides a dynamic, 100% Tkinter-based Air Keyboard.
Detects index-finger hover over keys, pinch-to-type execution via PyAutoGUI,
debounce protection to prevent repeated keys, and a live text buffer display.
"""

import time
import tkinter as tk
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False


# Keyboard Layout Definitions
ROW_NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "="]
ROW_NUMBERS_SHIFT = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+"]

ROW_QWERTY = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
ROW_ASDF = ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'"]
ROW_ZXCV = ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Back"]
ROW_BOTTOM = ["Caps", "Space", "Enter", "Clear"]


class VirtualAirKeyboard:
    """
    Dynamic Tkinter virtual keyboard capable of being operated via hand gestures.
    """

    def __init__(self, parent_container: tk.Widget, debounce_time: float = 0.50):
        self.parent = parent_container
        self.debounce_time = debounce_time

        self.shift_active: bool = False
        self.caps_lock: bool = False
        self.last_typed_time: float = 0.0
        self.is_pinch_active: bool = False

        self.hovered_key: Optional[str] = None
        self.key_buttons: Dict[str, tk.Label] = {}
        self.key_rects: Dict[str, Tuple[float, float, float, float]] = {}  # Normalized (x1, y1, x2, y2)

        self.typed_text_var = tk.StringVar(value="")

        self.frame = tk.Frame(self.parent, bg="#0f172a", bd=2, relief="groove")
        self._build_ui()

    def _build_ui(self):
        """Constructs the dark theme virtual keyboard layout dynamically."""
        # Top text preview bar
        preview_frame = tk.Frame(self.frame, bg="#1e293b", padx=6, pady=4)
        preview_frame.pack(fill="x", padx=6, pady=(6, 4))

        tk.Label(
            preview_frame,
            text="Typed Text:",
            font=("Segoe UI", 10, "bold"),
            fg="#94a3b8",
            bg="#1e293b"
        ).pack(side="left", padx=4)

        self.preview_label = tk.Label(
            preview_frame,
            textvariable=self.typed_text_var,
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            anchor="w",
            padx=8,
            relief="sunken"
        )
        self.preview_label.pack(side="left", fill="x", expand=True, padx=4)

        # Keys Grid Frame
        self.keys_frame = tk.Frame(self.frame, bg="#0f172a", padx=4, pady=4)
        self.keys_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.layout_rows = [
            ROW_NUMBERS,
            ROW_QWERTY,
            ROW_ASDF,
            ROW_ZXCV,
            ROW_BOTTOM
        ]

        # Dynamically build rows
        for r_idx, row in enumerate(self.layout_rows):
            row_frame = tk.Frame(self.keys_frame, bg="#0f172a")
            row_frame.pack(fill="x", expand=True, pady=2)

            for key in row:
                width = 3
                if key in ("Shift", "Caps", "Back", "Enter"):
                    width = 6
                elif key == "Space":
                    width = 16
                elif key == "Clear":
                    width = 6

                btn = tk.Label(
                    row_frame,
                    text=key,
                    font=("Segoe UI", 10, "bold"),
                    fg="#f8fafc",
                    bg="#1e293b",
                    width=width,
                    height=1,
                    relief="raised",
                    bd=1,
                    padx=2,
                    pady=4
                )
                btn.pack(side="left", fill="both", expand=True, padx=2)
                self.key_buttons[key] = btn

        # Calculate normalized key layout geometry on idle
        self.parent.after(200, self._calculate_key_geometries)

    def _calculate_key_geometries(self):
        """
        Computes the relative normalized coordinates (0..1) of each key
        inside the keyboard keys_frame to enable finger hit-testing.
        """
        try:
            total_w = self.keys_frame.winfo_width()
            total_h = self.keys_frame.winfo_height()

            if total_w < 50 or total_h < 50:
                # Retry after UI layout stabilizes
                self.parent.after(250, self._calculate_key_geometries)
                return

            frame_x = self.keys_frame.winfo_rootx()
            frame_y = self.keys_frame.winfo_rooty()

            for key, widget in self.key_buttons.items():
                wx = widget.winfo_rootx() - frame_x
                wy = widget.winfo_rooty() - frame_y
                ww = widget.winfo_width()
                wh = widget.winfo_height()

                norm_x1 = wx / total_w
                norm_y1 = wy / total_h
                norm_x2 = (wx + ww) / total_w
                norm_y2 = (wy + wh) / total_h

                self.key_rects[key] = (norm_x1, norm_y1, norm_x2, norm_y2)
        except Exception:
            pass

    def update_finger_hover(self, norm_x: float, norm_y: float) -> Optional[str]:
        """
        Takes normalized index-finger position [0..1] mapped to the keyboard area,
        applies comfortable margin mapping, finds which key is hovered, and highlights it.
        """
        if not self.key_rects:
            self._calculate_key_geometries()

        # Map active hand frame [0.12..0.88, 0.18..0.85] to [0..1]
        margin_x = 0.12
        margin_y = 0.18
        clamped_x = max(margin_x, min(1.0 - margin_x, norm_x))
        clamped_y = max(margin_y, min(1.0 - margin_y, norm_y))
        mapped_x = (clamped_x - margin_x) / (1.0 - 2 * margin_x)
        mapped_y = (clamped_y - margin_y) / (1.0 - 2 * margin_y)

        matched_key = None
        for key, (x1, y1, x2, y2) in self.key_rects.items():
            if x1 <= mapped_x <= x2 and y1 <= mapped_y <= y2:
                matched_key = key
                break

        # If hover state changed, update colors
        if matched_key != self.hovered_key:
            # Restore previous key color
            if self.hovered_key and self.hovered_key in self.key_buttons:
                self._reset_key_color(self.hovered_key)

            # Highlight newly hovered key
            self.hovered_key = matched_key
            if self.hovered_key and self.hovered_key in self.key_buttons:
                self.key_buttons[self.hovered_key].configure(
                    bg="#06b6d4",  # Glowing cyan
                    fg="#0f172a"
                )

        return self.hovered_key

    def _reset_key_color(self, key: str):
        """Restores key's normal color based on state."""
        if key not in self.key_buttons:
            return

        if key == "Caps" and self.caps_lock:
            self.key_buttons[key].configure(bg="#eab308", fg="#0f172a")
        elif key == "Shift" and self.shift_active:
            self.key_buttons[key].configure(bg="#eab308", fg="#0f172a")
        else:
            self.key_buttons[key].configure(bg="#1e293b", fg="#f8fafc")

    def execute_pinch_typing(self, is_pinch: bool) -> Optional[str]:
        """
        Handles pinch gesture to type the currently hovered key.
        Prevents key repetition (HHHHHH) by requiring pinch release and cooldown.
        """
        now = time.time()

        if not is_pinch:
            # Pinch released: allow typing next key
            self.is_pinch_active = False
            return None

        # Pinch is currently held down
        if self.is_pinch_active:
            # User is holding pinch, do not repeat
            return None

        if (now - self.last_typed_time) < self.debounce_time:
            # Cooldown active
            return None

        if self.hovered_key is None:
            return None

        key_to_press = self.hovered_key
        self.is_pinch_active = True
        self.last_typed_time = now

        # Flash key in emerald green to indicate successful selection
        if key_to_press in self.key_buttons:
            self.key_buttons[key_to_press].configure(bg="#10b981", fg="#ffffff")
            self.parent.after(180, lambda: self._reset_key_color(key_to_press))

        self._type_key(key_to_press)
        return key_to_press

    def _type_key(self, key: str):
        """Dispatches key to system via PyAutoGUI and updates local text preview."""
        current_text = self.typed_text_var.get()

        if key == "Space":
            self.typed_text_var.set(current_text + " ")
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.press("space")
                except Exception:
                    pass

        elif key in ("Back", "Backspace"):
            if len(current_text) > 0:
                self.typed_text_var.set(current_text[:-1])
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.press("backspace")
                except Exception:
                    pass

        elif key == "Enter":
            self.typed_text_var.set(current_text + "\n")
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.press("enter")
                except Exception:
                    pass

        elif key == "Clear":
            self.typed_text_var.set("")

        elif key == "Shift":
            self.shift_active = not self.shift_active
            self._update_keyboard_characters()

        elif key == "Caps":
            self.caps_lock = not self.caps_lock
            self._update_keyboard_characters()

        else:
            # Standard character
            char = key
            # Shift or Caps logic
            if self.shift_active or self.caps_lock:
                char = char.upper()
            else:
                char = char.lower()

            self.typed_text_var.set(current_text + char)
            if PYAUTOGUI_AVAILABLE:
                try:
                    pyautogui.write(char)
                except Exception:
                    pass

            # One-shot shift toggle resets after character
            if self.shift_active:
                self.shift_active = False
                self._update_keyboard_characters()

    def _update_keyboard_characters(self):
        """Updates key text labels based on Shift and Caps Lock states."""
        for key, btn in self.key_buttons.items():
            if key in ("Shift", "Caps", "Space", "Back", "Enter", "Clear"):
                self._reset_key_color(key)
                continue

            if key in ROW_NUMBERS and self.shift_active:
                idx = ROW_NUMBERS.index(key)
                btn.configure(text=ROW_NUMBERS_SHIFT[idx])
            elif key in ROW_NUMBERS and not self.shift_active:
                btn.configure(text=key)
            else:
                if self.shift_active or self.caps_lock:
                    btn.configure(text=key.upper())
                else:
                    btn.configure(text=key.lower())
            self._reset_key_color(key)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)
        self.parent.after(200, self._calculate_key_geometries)

    def pack_forget(self):
        self.frame.pack_forget()

"""
GestoControl - Main Graphical User Interface
100% Pure Python Tkinter Desktop Application.
Features a modern dark UI, live OpenCV/MediaPipe camera preview with HUD,
real-time status telemetry, mode switching, virtual air keyboard,
settings configuration dialog, and safety controls.
"""

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

from brightness_controller import BrightnessController
from gesture_engine import (
    GestureEngine,
    GESTURE_FIST,
    GESTURE_NONE,
    GESTURE_OPEN_PALM,
    GESTURE_PINCH,
    GESTURE_POINT,
    GESTURE_SWIPE_DOWN,
    GESTURE_SWIPE_LEFT,
    GESTURE_SWIPE_RIGHT,
    GESTURE_SWIPE_UP,
    GESTURE_TWO_FINGER,
)
from hand_tracking import HandTracker
from keyboard_controller import VirtualAirKeyboard
from media_controller import MediaController
from mouse_controller import MouseController
from reels_controller import ReelsController
from settings import SettingsManager
from volume_controller import VolumeController


class GestoControlUI:
    """Main Desktop Application Window for GestoControl."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("GestoControl - Hand Gesture Laptop Controller")
        self.root.geometry("1180x820")
        self.root.minsize(1050, 750)
        self.root.configure(bg="#0b0f19")

        # Load configuration
        self.settings_mgr = SettingsManager()
        self.cfg = self.settings_mgr.settings

        # State Variables
        self.running: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_mode: str = self.cfg.get("current_mode", "REELS")
        self.gesture_control_enabled: bool = self.cfg.get("gesture_control_enabled", True)

        self.last_action_text: str = "System Ready"
        self.last_gesture_text: str = GESTURE_NONE
        self.confidence_val: float = 0.0
        self.fps: float = 0.0
        self.prev_frame_time: float = time.time()

        # Telemetry Display Variables (Tkinter StringVars)
        self.var_hand_status = tk.StringVar(value="NOT DETECTED")
        self.var_mode = tk.StringVar(value=self.current_mode)
        self.var_gesture = tk.StringVar(value="NONE")
        self.var_action = tk.StringVar(value="System Ready")
        self.var_brightness = tk.StringVar(value="50%")
        self.var_volume = tk.StringVar(value="50%")
        self.var_fps = tk.StringVar(value="0 FPS")
        self.var_master_state = tk.StringVar(
            value="GESTURE CONTROL: ON" if self.gesture_control_enabled else "GESTURE CONTROL: OFF (SAFETY)"
        )

        # Initialize Subsystem Controllers
        self.hand_tracker = HandTracker(
            max_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70
        )

        self.gesture_engine = GestureEngine(
            pinch_threshold=float(self.cfg.get("pinch_threshold", 0.055)),
            two_finger_threshold=float(self.cfg.get("two_finger_threshold", 0.065)),
            swipe_min_distance=float(self.cfg.get("swipe_min_distance", 0.10)),
            swipe_cooldown=float(self.cfg.get("swipe_cooldown", 0.65)),
            action_cooldown=float(self.cfg.get("action_cooldown", 0.45))
        )

        self.brightness_ctrl = BrightnessController(
            min_distance=float(self.cfg.get("brightness_min_dist", 0.03)),
            max_distance=float(self.cfg.get("brightness_max_dist", 0.26)),
            smoothing_factor=float(self.cfg.get("brightness_smoothing", 0.30))
        )

        self.volume_ctrl = VolumeController(
            min_distance=float(self.cfg.get("volume_min_dist", 0.03)),
            max_distance=float(self.cfg.get("volume_max_dist", 0.26)),
            smoothing_factor=float(self.cfg.get("volume_smoothing", 0.30))
        )
        self.var_volume.set(f"{self.volume_ctrl.get_volume_percentage()}%")

        self.mouse_ctrl = MouseController(
            sensitivity=float(self.cfg.get("mouse_sensitivity", 1.6)),
            smoothing=float(self.cfg.get("mouse_smoothing", 0.35)),
            deadzone=float(self.cfg.get("mouse_deadzone", 0.004))
        )

        self.reels_ctrl = ReelsController(
            key_bindings=self.cfg.get("reels_keys", {})
        )

        self.media_ctrl = MediaController(
            key_bindings=self.cfg.get("media_keys", {})
        )

        # Build Interface
        self._build_header()
        self._build_main_layout()
        self._highlight_active_mode_button()

        # Handle window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_header(self):
        """Builds top banner and master safety toggle."""
        header_frame = tk.Frame(self.root, bg="#111827", pady=10, padx=20, bd=1, relief="ridge")
        header_frame.pack(fill="x")

        title_box = tk.Frame(header_frame, bg="#111827")
        title_box.pack(side="left")

        tk.Label(
            title_box,
            text="GESTOCONTROL",
            font=("Segoe UI", 20, "bold"),
            fg="#38bdf8",
            bg="#111827"
        ).pack(side="left")

        tk.Label(
            title_box,
            text=" | Hand Gesture Laptop Controller",
            font=("Segoe UI", 12),
            fg="#94a3b8",
            bg="#111827"
        ).pack(side="left", padx=8, pady=(4, 0))

        # Master Gesture Control Safety Toggle Button
        master_btn_color = "#10b981" if self.gesture_control_enabled else "#ef4444"
        self.btn_master_toggle = tk.Button(
            header_frame,
            textvariable=self.var_master_state,
            command=self.toggle_master_gesture_control,
            font=("Segoe UI", 10, "bold"),
            bg=master_btn_color,
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        )
        self.btn_master_toggle.pack(side="right")

    def _build_main_layout(self):
        """Constructs split-pane layout: Camera View on left, Dashboard Controls on right."""
        body_frame = tk.Frame(self.root, bg="#0b0f19")
        body_frame.pack(fill="both", expand=True, padx=16, pady=12)

        # LEFT PANE: Video Canvas & Air Keyboard Container
        self.left_pane = tk.Frame(body_frame, bg="#0f172a", bd=1, relief="solid")
        self.left_pane.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Video Preview Canvas
        self.video_container = tk.Frame(self.left_pane, bg="#000000")
        self.video_container.pack(fill="both", expand=True)

        self.video_label = tk.Label(
            self.video_container,
            text="Camera is stopped.\nClick [ START CAMERA ] to begin.",
            font=("Segoe UI", 14),
            fg="#64748b",
            bg="#000000",
            justify="center"
        )
        self.video_label.pack(fill="both", expand=True)

        # Virtual Keyboard Container (attached under video canvas)
        self.keyboard_container = tk.Frame(self.left_pane, bg="#0f172a")
        self.virtual_keyboard = VirtualAirKeyboard(
            parent_container=self.keyboard_container,
            debounce_time=float(self.cfg.get("keyboard_debounce", 0.50))
        )
        # Packed conditionally when Air Keyboard mode is chosen
        if self.current_mode == "AIR KEYBOARD":
            self.keyboard_container.pack(fill="x", side="bottom", padx=8, pady=8)
            self.virtual_keyboard.pack(fill="x", expand=True)

        # RIGHT PANE: Telemetry Cards & Control Buttons
        right_pane = tk.Frame(body_frame, bg="#111827", width=330, bd=1, relief="solid")
        right_pane.pack(side="right", fill="y", padx=(10, 0))
        right_pane.pack_propagate(False)

        # Telemetry Card
        telemetry_frame = tk.LabelFrame(
            right_pane,
            text=" REAL-TIME STATUS ",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=14,
            pady=12,
            bd=1,
            relief="groove"
        )
        telemetry_frame.pack(fill="x", padx=12, pady=(12, 8))

        # Helper to create telemetry rows
        def create_telemetry_row(parent, label_text: str, text_var: tk.StringVar, value_color="#f8fafc"):
            row = tk.Frame(parent, bg="#1e293b")
            row.pack(fill="x", pady=3)
            tk.Label(
                row,
                text=label_text,
                font=("Segoe UI", 9, "bold"),
                fg="#94a3b8",
                bg="#1e293b",
                width=14,
                anchor="w"
            ).pack(side="left")
            val_lbl = tk.Label(
                row,
                textvariable=text_var,
                font=("Segoe UI", 9, "bold"),
                fg=value_color,
                bg="#1e293b",
                anchor="w"
            )
            val_lbl.pack(side="left", fill="x", expand=True)
            return val_lbl

        self.lbl_hand_status = create_telemetry_row(telemetry_frame, "Hand Status:", self.var_hand_status, "#ef4444")
        create_telemetry_row(telemetry_frame, "Current Mode:", self.var_mode, "#38bdf8")
        create_telemetry_row(telemetry_frame, "Current Gesture:", self.var_gesture, "#facc15")
        create_telemetry_row(telemetry_frame, "Current Action:", self.var_action, "#4ade80")
        create_telemetry_row(telemetry_frame, "Brightness:", self.var_brightness, "#f472b6")
        create_telemetry_row(telemetry_frame, "System Volume:", self.var_volume, "#38bdf8")
        create_telemetry_row(telemetry_frame, "Frame Rate:", self.var_fps, "#a78bfa")

        # Visual Brightness Bar
        prog_frame = tk.Frame(telemetry_frame, bg="#1e293b")
        prog_frame.pack(fill="x", pady=(6, 0))
        tk.Label(prog_frame, text="Brightness / Volume Gauge:", font=("Segoe UI", 8), fg="#94a3b8", bg="#1e293b", anchor="w").pack(fill="x")
        self.brightness_bar = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.brightness_bar["value"] = 50
        self.brightness_bar.pack(fill="x")

        # Operational Mode Selector Section
        mode_section = tk.LabelFrame(
            right_pane,
            text=" SELECT MODE ",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=10,
            pady=10,
            bd=1,
            relief="groove"
        )
        mode_section.pack(fill="x", padx=12, pady=6)

        self.mode_buttons = {}
        modes = [
            ("VOLUME", "🔊 Volume Mode"),
            ("BRIGHTNESS", "🔆 Brightness Mode"),
            ("REELS", "📱 Reels / Shorts Mode"),
            ("AIR MOUSE", "🖱️ Air Mouse Mode"),
            ("AIR KEYBOARD", "⌨️ Air Keyboard Mode")
        ]

        for mode_key, mode_title in modes:
            btn = tk.Button(
                mode_section,
                text=mode_title,
                command=lambda m=mode_key: self.set_mode(m),
                font=("Segoe UI", 9, "bold"),
                bg="#334155",
                fg="#f8fafc",
                activebackground="#0284c7",
                activeforeground="#ffffff",
                bd=0,
                pady=6,
                cursor="hand2"
            )
            btn.pack(fill="x", pady=2)
            self.mode_buttons[mode_key] = btn

        # Camera & System Operations Section
        ops_section = tk.LabelFrame(
            right_pane,
            text=" SYSTEM CONTROLS ",
            font=("Segoe UI", 10, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=10,
            pady=10,
            bd=1,
            relief="groove"
        )
        ops_section.pack(fill="both", expand=True, padx=12, pady=(6, 12))

        self.btn_start = tk.Button(
            ops_section,
            text="▶  START CAMERA",
            command=self.start_camera,
            font=("Segoe UI", 10, "bold"),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            bd=0,
            pady=7,
            cursor="hand2"
        )
        self.btn_start.pack(fill="x", pady=3)

        self.btn_stop = tk.Button(
            ops_section,
            text="⏹  STOP CAMERA",
            command=self.stop_camera,
            font=("Segoe UI", 10, "bold"),
            bg="#64748b",
            fg="#ffffff",
            activebackground="#475569",
            bd=0,
            pady=7,
            cursor="hand2",
            state="disabled"
        )
        self.btn_stop.pack(fill="x", pady=3)

        self.btn_settings = tk.Button(
            ops_section,
            text="⚙  SETTINGS",
            command=self.open_settings_window,
            font=("Segoe UI", 10, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            bd=0,
            pady=7,
            cursor="hand2"
        )
        self.btn_settings.pack(fill="x", pady=3)

        btn_exit = tk.Button(
            ops_section,
            text="✕  EXIT",
            command=self.on_close,
            font=("Segoe UI", 10, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            bd=0,
            pady=7,
            cursor="hand2"
        )
        btn_exit.pack(fill="x", side="bottom", pady=3)

    def _highlight_active_mode_button(self):
        """Highlights the active mode button with vibrant color."""
        for m_key, btn in self.mode_buttons.items():
            if m_key == self.current_mode:
                btn.configure(bg="#0284c7", fg="#ffffff")
            else:
                btn.configure(bg="#334155", fg="#f8fafc")

    def toggle_master_gesture_control(self):
        """Toggles the safety master switch on/off."""
        self.gesture_control_enabled = not self.gesture_control_enabled
        self.cfg["gesture_control_enabled"] = self.gesture_control_enabled
        self.settings_mgr.set("gesture_control_enabled", self.gesture_control_enabled)
        self.settings_mgr.save()

        if self.gesture_control_enabled:
            self.var_master_state.set("GESTURE CONTROL: ON")
            self.btn_master_toggle.configure(bg="#10b981", activebackground="#059669")
            self.var_action.set("Gestures Enabled")
        else:
            self.var_master_state.set("GESTURE CONTROL: OFF (SAFETY)")
            self.btn_master_toggle.configure(bg="#ef4444", activebackground="#dc2626")
            self.var_action.set("Safety Pause Active")

    def set_mode(self, mode: str):
        """Switches the active interaction mode."""
        self.current_mode = mode
        self.var_mode.set(mode)
        self.cfg["current_mode"] = mode
        self.settings_mgr.set("current_mode", mode)
        self.settings_mgr.save()

        self._highlight_active_mode_button()

        # Handle Virtual Keyboard display visibility
        if mode == "AIR KEYBOARD":
            self.keyboard_container.pack(fill="x", side="bottom", padx=8, pady=8)
            self.virtual_keyboard.pack(fill="x", expand=True)
        else:
            self.virtual_keyboard.pack_forget()
            self.keyboard_container.pack_forget()

        self.var_action.set(f"Switched to {mode}")

    def start_camera(self):
        """Opens webcam and begins processing loop."""
        if self.running:
            return

        camera_idx = int(self.cfg.get("camera_index", 0))
        try:
            self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                # Try default backend if DirectShow fails
                self.cap = cv2.VideoCapture(camera_idx)
        except Exception as err:
            messagebox.showerror(
                "Camera Error",
                f"Failed to initialize webcam (Index {camera_idx}).\n\nError: {err}\n\nPlease connect a webcam and try again."
            )
            return

        if not self.cap.isOpened():
            messagebox.showerror(
                "Webcam Not Found",
                f"Could not connect to webcam on index {camera_idx}.\n\nPlease ensure your camera is plugged in, not used by another application, and permissions are granted."
            )
            return

        # Configure camera resolution
        w = int(self.cfg.get("camera_width", 640))
        h = int(self.cfg.get("camera_height", 480))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self.running = True
        self.btn_start.configure(state="disabled", bg="#064e3b")
        self.btn_stop.configure(state="normal", bg="#ef4444")
        self.var_action.set("Camera Running")

        # Launch video processing loop
        self._camera_loop()

    def stop_camera(self):
        """Stops camera and releases resources."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.configure(
            image="",
            text="Camera is stopped.\nClick [ START CAMERA ] to begin."
        )
        self.var_hand_status.set("NOT DETECTED")
        self.lbl_hand_status.configure(fg="#ef4444")
        self.var_gesture.set("NONE")
        self.var_fps.set("0 FPS")
        self.btn_start.configure(state="normal", bg="#10b981")
        self.btn_stop.configure(state="disabled", bg="#64748b")
        self.var_action.set("Camera Stopped")

    def _camera_loop(self):
        """Primary real-time processing loop for frame capture and gesture dispatch."""
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Brief delay before retrying
            self.root.after(20, self._camera_loop)
            return

        # Horizontal mirror flip so hand movements match user intuition
        if self.cfg.get("flip_camera", True):
            frame = cv2.flip(frame, 1)

        # FPS calculation
        now = time.time()
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.var_fps.set(f"{int(round(self.fps))} FPS")

        # Hand Detection & Tracking
        hand_data = self.hand_tracker.process(frame)

        if hand_data is not None:
            self.var_hand_status.set("DETECTED")
            self.lbl_hand_status.configure(fg="#22c55e")

            # Draw visual skeleton on preview
            self.hand_tracker.draw_skeleton(frame, hand_data)

            # Process gestures through priority engine
            detected_gesture, meta = self.gesture_engine.detect_gesture(
                hand_data=hand_data,
                current_mode=self.current_mode
            )
            self.var_gesture.set(detected_gesture)

            # Dispatch action according to active mode (if master gesture control is enabled)
            if self.gesture_control_enabled:
                self._dispatch_mode_action(detected_gesture, hand_data, meta)
            else:
                self.var_action.set("Safety Pause (Off)")

        else:
            self.var_hand_status.set("NOT DETECTED")
            self.lbl_hand_status.configure(fg="#ef4444")
            self.var_gesture.set(GESTURE_NONE)
            # Reset gesture engine trajectory
            self.gesture_engine.detect_gesture(None, self.current_mode)
            if self.current_mode == "AIR MOUSE":
                self.mouse_ctrl.release_pinch()
                self.mouse_ctrl.release_two_finger()

        # Render Camera HUD Overlay
        self._render_camera_hud(frame, hand_data is not None)

        # Convert frame for Tkinter Label display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Scale image to fit container while maintaining aspect ratio
        container_w = max(320, self.video_container.winfo_width())
        container_h = max(240, self.video_container.winfo_height())
        pil_img.thumbnail((container_w, container_h), Image.Resampling.BILINEAR)

        photo = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        # Schedule next frame (~30-60 FPS)
        self.root.after(10, self._camera_loop)

    def _dispatch_mode_action(self, gesture: str, hand_data, meta: dict):
        """Dispatches detected hand actions to the appropriate controller."""
        # 1. VOLUME MODE (Increase / Decrease System Volume)
        if self.current_mode == "VOLUME":
            if gesture == GESTURE_FIST:
                is_muted = self.volume_ctrl.toggle_mute()
                self.var_action.set("MUTED" if is_muted else "UNMUTED")
                self.gesture_engine.record_action_executed()
            elif gesture == GESTURE_SWIPE_UP:
                v = self.volume_ctrl.increase_volume(step=5)
                self.var_volume.set(f"{v}%")
                self.brightness_bar["value"] = v
                self.var_action.set(f"Vol Up (+5%): {v}%")
                self.gesture_engine.record_action_executed()
            elif gesture == GESTURE_SWIPE_DOWN:
                v = self.volume_ctrl.decrease_volume(step=5)
                self.var_volume.set(f"{v}%")
                self.brightness_bar["value"] = v
                self.var_action.set(f"Vol Down (-5%): {v}%")
                self.gesture_engine.record_action_executed()
            else:
                # Continuous distance tracking between Thumb tip & Index tip
                v_val = self.volume_ctrl.update_from_landmarks(
                    thumb_tip=hand_data.thumb_tip,
                    index_tip=hand_data.index_tip,
                    apply_hardware=True
                )
                self.var_volume.set(f"{v_val}%")
                self.brightness_bar["value"] = v_val
                self.var_action.set(f"Volume: {v_val}%")

        # 2. BRIGHTNESS MODE
        elif self.current_mode == "BRIGHTNESS":
            b_val = self.brightness_ctrl.update_from_landmarks(
                thumb_tip=hand_data.thumb_tip,
                index_tip=hand_data.index_tip,
                apply_hardware=True
            )
            self.var_brightness.set(f"{b_val}%")
            self.brightness_bar["value"] = b_val
            self.var_action.set(f"Adjusting: {b_val}%")

        # 2. REELS / SHORTS MODE
        elif self.current_mode == "REELS":
            if gesture != GESTURE_NONE:
                executed, act_desc = self.reels_ctrl.execute_gesture(gesture)
                if executed:
                    self.var_action.set(act_desc)
                    self.gesture_engine.record_action_executed()

        # 3. AIR MOUSE MODE
        elif self.current_mode == "AIR MOUSE":
            if gesture == GESTURE_OPEN_PALM:
                # Open palm stops/freezes mouse cursor movement
                self.var_action.set("CURSOR FROZEN (PALM)")
            elif gesture == GESTURE_TWO_FINGER:
                if self.mouse_ctrl.right_click():
                    self.var_action.set("RIGHT CLICK")
                    self.gesture_engine.record_action_executed()
            elif gesture == GESTURE_PINCH:
                if self.mouse_ctrl.left_click():
                    self.var_action.set("LEFT CLICK")
                    self.gesture_engine.record_action_executed()
            else:
                # Normal index finger cursor gliding
                self.mouse_ctrl.release_pinch()
                self.mouse_ctrl.release_two_finger()
                idx_x, idx_y = hand_data.index_tip
                sx, sy = self.mouse_ctrl.move(idx_x, idx_y)
                self.var_action.set(f"Cursor ({sx}, {sy})")

        # 4. AIR KEYBOARD MODE
        elif self.current_mode == "AIR KEYBOARD":
            idx_x, idx_y = hand_data.index_tip
            hovered = self.virtual_keyboard.update_finger_hover(idx_x, idx_y)

            is_pinch = (gesture == GESTURE_PINCH)
            typed_key = self.virtual_keyboard.execute_pinch_typing(is_pinch)

            if typed_key:
                self.var_action.set(f"Typed: '{typed_key}'")
                self.gesture_engine.record_action_executed()
            elif hovered:
                self.var_action.set(f"Hover: '{hovered}'")
            else:
                self.var_action.set("Air Typing Active")

    def _render_camera_hud(self, frame: np.ndarray, hand_detected: bool):
        """Draws aesthetic status HUD directly onto the OpenCV preview."""
        h, w, _ = frame.shape

        # Semi-transparent top HUD banner
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (w, 42), (15, 23, 42), -1)
        cv2.addWeighted(hud_bg, 0.70, frame, 0.30, 0, frame)

        # Hand detection badge
        badge_text = "HAND: DETECTED" if hand_detected else "HAND: NOT DETECTED"
        badge_color = (0, 255, 127) if hand_detected else (0, 0, 255)
        cv2.circle(frame, (18, 21), 6, badge_color, -1, cv2.LINE_AA)
        cv2.putText(frame, badge_text, (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)

        # Mode banner in center
        mode_text = f"MODE: {self.current_mode}"
        cv2.putText(frame, mode_text, (int(w / 2 - 60), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2, cv2.LINE_AA)

        # FPS badge on right
        fps_text = f"{int(round(self.fps))} FPS"
        cv2.putText(frame, fps_text, (w - 85, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Gesture & Action banner on bottom
        bottom_bg = frame.copy()
        cv2.rectangle(bottom_bg, (0, h - 36), (w, h), (15, 23, 42), -1)
        cv2.addWeighted(bottom_bg, 0.70, frame, 0.30, 0, frame)

        gest_info = f"GESTURE: {self.var_gesture.get()}  |  ACTION: {self.var_action.get()}"
        cv2.putText(frame, gest_info, (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    def open_settings_window(self):
        """Displays interactive Tkinter modal dialog to customize sensitivities and preferences."""
        dlg = tk.Toplevel(self.root)
        dlg.title("GestoControl - Settings")
        dlg.geometry("520x620")
        dlg.minsize(480, 560)
        dlg.configure(bg="#0f172a")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(
            dlg,
            text="Application Settings",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#0f172a"
        ).pack(pady=(14, 8))

        form = tk.Frame(dlg, bg="#0f172a", padx=20)
        form.pack(fill="both", expand=True)

        def add_slider_row(parent, label_text: str, from_val: float, to_val: float, init_val: float, resolution: float = 0.01):
            row = tk.Frame(parent, bg="#0f172a")
            row.pack(fill="x", pady=6)
            tk.Label(
                row,
                text=label_text,
                font=("Segoe UI", 9, "bold"),
                fg="#cbd5e1",
                bg="#0f172a",
                width=22,
                anchor="w"
            ).pack(side="left")
            scale = tk.Scale(
                row,
                from_=from_val,
                to=to_val,
                resolution=resolution,
                orient="horizontal",
                bg="#1e293b",
                fg="#f8fafc",
                highlightthickness=0,
                bd=0,
                length=220
            )
            scale.set(init_val)
            scale.pack(side="right", fill="x", expand=True)
            return scale

        # Camera index selection
        cam_row = tk.Frame(form, bg="#0f172a")
        cam_row.pack(fill="x", pady=6)
        tk.Label(
            cam_row,
            text="Camera Index:",
            font=("Segoe UI", 9, "bold"),
            fg="#cbd5e1",
            bg="#0f172a",
            width=22,
            anchor="w"
        ).pack(side="left")
        cam_entry = tk.Spinbox(cam_row, from_=0, to=5, width=6, font=("Segoe UI", 10))
        cam_entry.delete(0, "end")
        cam_entry.insert(0, str(self.cfg.get("camera_index", 0)))
        cam_entry.pack(side="left")

        # Threshold Sliders
        scale_swipe = add_slider_row(form, "Swipe Sensitivity (Dist):", 0.05, 0.30, float(self.cfg.get("swipe_min_distance", 0.10)))
        scale_pinch = add_slider_row(form, "Pinch Sensitivity (Dist):", 0.02, 0.10, float(self.cfg.get("pinch_threshold", 0.055)))
        scale_mouse_sens = add_slider_row(form, "Mouse Sensitivity:", 0.5, 3.5, float(self.cfg.get("mouse_sensitivity", 1.6)), resolution=0.1)
        scale_mouse_smooth = add_slider_row(form, "Mouse Smoothing (EMA):", 0.10, 0.80, float(self.cfg.get("mouse_smoothing", 0.35)))
        scale_bright_smooth = add_slider_row(form, "Brightness Sensitivity:", 0.10, 0.70, float(self.cfg.get("brightness_smoothing", 0.30)))
        scale_volume_smooth = add_slider_row(form, "Volume Sensitivity/Smoothing:", 0.10, 0.70, float(self.cfg.get("volume_smoothing", 0.30)))
        scale_cooldown = add_slider_row(form, "Gesture Cooldown (sec):", 0.20, 1.50, float(self.cfg.get("swipe_cooldown", 0.65)))

        # Action Buttons
        btn_box = tk.Frame(dlg, bg="#0f172a", pady=16)
        btn_box.pack(fill="x")

        def save_and_apply():
            try:
                self.cfg["camera_index"] = int(cam_entry.get())
                self.cfg["swipe_min_distance"] = float(scale_swipe.get())
                self.cfg["pinch_threshold"] = float(scale_pinch.get())
                self.cfg["mouse_sensitivity"] = float(scale_mouse_sens.get())
                self.cfg["mouse_smoothing"] = float(scale_mouse_smooth.get())
                self.cfg["brightness_smoothing"] = float(scale_bright_smooth.get())
                self.cfg["volume_smoothing"] = float(scale_volume_smooth.get())
                self.cfg["swipe_cooldown"] = float(scale_cooldown.get())

                # Update running controllers
                self.gesture_engine.update_settings(
                    pinch_threshold=self.cfg["pinch_threshold"],
                    swipe_min_distance=self.cfg["swipe_min_distance"],
                    swipe_cooldown=self.cfg["swipe_cooldown"]
                )
                self.mouse_ctrl.sensitivity = self.cfg["mouse_sensitivity"]
                self.mouse_ctrl.smoothing = self.cfg["mouse_smoothing"]
                self.brightness_ctrl.smoothing_factor = self.cfg["brightness_smoothing"]
                self.volume_ctrl.smoothing_factor = self.cfg["volume_smoothing"]

                self.settings_mgr.save()
                messagebox.showinfo("Settings Saved", "Settings successfully updated and saved locally!", parent=dlg)
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}", parent=dlg)

        def reset_defaults():
            if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?", parent=dlg):
                self.settings_mgr.reset_to_defaults()
                self.cfg = self.settings_mgr.settings
                dlg.destroy()
                self.open_settings_window()

        tk.Button(
            btn_box,
            text="SAVE SETTINGS",
            command=save_and_apply,
            bg="#10b981",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="left", padx=16)

        tk.Button(
            btn_box,
            text="RESET DEFAULTS",
            command=reset_defaults,
            bg="#64748b",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="left", padx=4)

        tk.Button(
            btn_box,
            text="CANCEL",
            command=dlg.destroy,
            bg="#334155",
            fg="#cbd5e1",
            font=("Segoe UI", 10),
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2"
        ).pack(side="right", padx=16)

    def on_close(self):
        """Clean shutdown handler."""
        self.stop_camera()
        if self.hand_tracker:
            self.hand_tracker.close()
        self.root.destroy()

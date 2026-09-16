# GestoControl – Python Hand Gesture Laptop Controller

A touchless, computer-vision-powered desktop controller built **100% in Python** using **OpenCV, MediaPipe, NumPy, PyAutoGUI, and Tkinter**. Control your laptop screen brightness, browse Instagram Reels & YouTube Shorts, glide your mouse cursor in mid-air, and type on an on-screen virtual keyboard without ever touching physical peripherals.

---

## 🌟 Key Features

1. **Webcam Hand Tracking (MediaPipe 21 Landmarks)**
   - Real-time hand landmark extraction and skeleton overlay visualization.
   - Computes palm center coordinates and dynamic hand bounding boxes.
   - Live detection badge (`Hand: DETECTED` / `Hand: NOT DETECTED`).

2. **Thumb + Index Distance Brightness Control**
   - Measures Euclidean distance between thumb tip (Landmark 4) and index fingertip (Landmark 8) with NumPy.
   - Dynamically maps distance to $0\% - 100\%$ display brightness with Exponential Moving Average (EMA) smoothing to eliminate flickering.
   - Integrates Windows DDC/CI and WMI interfaces with graceful virtual fallback for external monitors.

3. **Multi-Frame Hand Swipe Detection**
   - Tracks palm center movement across consecutive frames over a sliding time window.
   - Distinguishes intentional directional swipes (`SWIPE UP`, `SWIPE DOWN`, `SWIPE LEFT`, `SWIPE RIGHT`) from static hand drift.
   - Equipped with configurable swipe distance thresholds, debouncing, and cooldown timers.

4. **Reels & Short Video Mode**
   - Touchless navigation for TikTok, Instagram Reels, YouTube Shorts, and Reddit.
   - Swipe Up $\rightarrow$ Next Reel / Video (`down` arrow / `pagedown`).
   - Swipe Down $\rightarrow$ Previous Reel (`up` arrow / `pageup`).
   - Open Palm $\rightarrow$ Play / Pause (`space`).
   - Fist $\rightarrow$ Pause / Mute (`k`).
   - Pinch $\rightarrow$ Like / Favorite (`l`).

5. **Air Mouse Mode**
   - High-precision cursor gliding tracking the index fingertip.
   - Exponential Moving Average (EMA) smoothing and dead-zone filtering to eliminate hand tremor.
   - Pinch (Thumb + Index) $\rightarrow$ Left Click (debounced).
   - Two-Finger Tap (Index + Middle extended together) $\rightarrow$ Right Click.
   - Open Palm $\rightarrow$ Safety freeze (pauses cursor movement).

6. **Virtual Air Keyboard Mode**
   - Dynamically rendered on-screen QWERTY keyboard built entirely with **Tkinter**.
   - Features number row, QWERTY letters, punctuation, Shift, Caps Lock, Space, Backspace, and Enter.
   - Index fingertip hover detection with glowing cyan highlight.
   - Thumb + Index pinch-to-type with debounce and pinch-release protection to prevent repeated keystrokes (`HHHHHH`).
   - Live typed text buffer preview.

7. **Dedicated Gesture Priority Engine**
   - Enforces a strict priority hierarchy to prevent gesture conflict:
     $$\text{PINCH} \succ \text{TWO FINGER} \succ \text{FIST} \succ \text{OPEN PALM} \succ \text{SWIPE} \succ \text{POINT}$$

8. **Master Safety Switch & Telemetry HUD**
   - Master toggle: `[ GESTURE CONTROL: ON / OFF ]`. When turned OFF, camera tracking and visual HUD stay active, but all laptop inputs are safely suspended.
   - Real-time on-screen HUD displays FPS, active mode, detected gesture, executed action, and brightness percentage.

9. **Interactive Settings Dialog**
   - Tkinter dialog to fine-tune camera index, swipe sensitivity, pinch threshold, mouse sensitivity, smoothing weights, and cooldown timers.
   - Automatically saves and loads configurations from local `settings.json`.

---

## 🏛️ Architecture & Pipeline

```text
       WEBCAM (OpenCV cv2.VideoCapture)
                     │
                     ▼
       MEDIAPIPE HANDS (21 3D Landmarks)
                     │
                     ▼
       NUMPY VECTOR & DISTANCE CALCULATIONS
                     │
                     ▼
      GESTURE RECOGNITION & PRIORITY ENGINE
        ├── Priority 1: Pinch (Thumb + Index)
        ├── Priority 2: Two Finger (Right Click)
        ├── Priority 3: Fist (Pause/Mute)
        ├── Priority 4: Open Palm (Freeze/Play)
        └── Priority 5: Multi-Frame Swipe Trajectory
                     │
                     ▼
              MODE DISPATCHER
        ├── Volume Mode        -> pycaw (Core Audio Windows API) / PyAutoGUI
        ├── Brightness Mode    -> screen-brightness-control / WMI
        ├── Reels Mode         -> PyAutoGUI navigation shortcuts
        ├── Air Mouse Mode     -> PyAutoGUI cursor & click events
        └── Air Keyboard Mode  -> Tkinter virtual keyboard & typing
                     │
                     ▼
           MODERN TKINTER DESKTOP GUI
        ├── Real-Time OpenCV Video HUD Canvas
        ├── Live Telemetry Dashboard (Brightness & Volume Gauges)
        ├── Dynamic Virtual Keyboard Dock
        └── Settings Modal Dialog
```

---

## 📁 Project Structure

```text
GestoControl/
│
├── main.py                  # Application entry point & dependency check
├── hand_tracking.py         # MediaPipe landmark extraction & skeleton renderer
├── gesture_engine.py        # Gesture classifier, trajectory tracker & priority engine
├── volume_controller.py     # Master system volume controller (continuous & discrete)
├── brightness_controller.py # Thumb-index distance mapping & screen brightness adjustment
├── mouse_controller.py      # Air mouse gliding, smoothing & click handler
├── keyboard_controller.py   # Pure Tkinter virtual keyboard & air-typing logic
├── reels_controller.py      # Configurable shortcuts for Reels/Shorts/TikTok
├── media_controller.py      # System media controls (play, pause, next, volume)
├── settings.py              # Configuration manager with settings.json persistence
├── ui.py                    # Main Tkinter desktop interface & settings dialog
├── requirements.txt         # Required Python packages
└── README.md                # Comprehensive documentation
```

---

## 🖐️ Gesture Reference Guide

| Gesture | Visual Description | Volume Mode | Brightness Mode | Reels Mode | Air Mouse Mode | Air Keyboard Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pinch** | Thumb tip + Index tip close | Continuous $0\% - 100\%$ adjust | Continuous $0\% - 100\%$ adjust | Like video (`l`) | Left Click | Type hovered key |
| **Two Finger** | Index + Middle extended | — | — | — | Right Click | — |
| **Swipe Up** | Rapid upward palm movement | Vol Up ($+5\%$) | — | Next Reel (`down`) | — | — |
| **Swipe Down** | Rapid downward palm movement | Vol Down ($-5\%$) | — | Previous Reel (`up`) | — | — |
| **Swipe Left** | Rapid leftward palm movement | — | — | Previous item | — | — |
| **Swipe Right** | Rapid rightward palm movement | — | — | Next item | — | — |
| **Open Palm** | All 5 fingers open wide | — | — | Play / Pause (`space`) | Freeze Cursor | — |
| **Fist** | All fingers curled into palm | Mute / Unmute Toggle | — | Pause / Mute (`k`) | — | — |
| **Point** | Only Index finger extended | — | — | — | Move Cursor | Hover over key |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Python 3.8 to 3.12** installed on Windows.
- A standard laptop webcam or USB external camera.

### 2. Install Dependencies
Open a terminal (Command Prompt or PowerShell) inside the `GestoControl` directory and run:

```bash
pip install -r requirements.txt
```

### 3. Launch GestoControl
Start the desktop application:

```bash
python main.py
```

---

## 📖 Step-by-Step Usage Guide

1. **Start the Camera**:
   - Click the green `[ START CAMERA ]` button.
   - Position your hand in front of the camera (about $0.5\text{m} - 1.0\text{m}$ distance).
   - Your hand skeleton and landmark points will appear with real-time HUD telemetry.

2. **Adjust Screen Brightness**:
   - Click `[ BRIGHTNESS MODE ]`.
   - Bring your thumb and index fingertip close together: screen brightness smoothly dims toward $0\%$.
   - Separate your thumb and index fingertip wide apart: screen brightness brightens up to $100\%$.

3. **Browse Instagram Reels / YouTube Shorts**:
   - Open your browser to YouTube Shorts, Instagram Reels, or TikTok.
   - Click `[ REELS MODE ]` in GestoControl.
   - Swipe your hand **Up** to skip to the next video.
   - Swipe your hand **Down** to return to the previous video.
   - Show an **Open Palm** to pause or resume playback.
   - Make a quick **Pinch** to like the reel.

4. **Operate the Air Mouse**:
   - Click `[ AIR MOUSE ]`.
   - Extend your index finger and move it in front of the camera: your mouse cursor will glide smoothly across the screen.
   - Pinch your thumb and index together to perform a **Left Click**.
   - Extend both your index and middle fingers together to perform a **Right Click**.
   - Show an **Open Palm** to temporarily pause mouse movement while repositioning your hand.

5. **Type with the Air Keyboard**:
   - Click `[ AIR KEYBOARD ]`.
   - The virtual keyboard dock will appear below the camera preview.
   - Move your index fingertip over any key: the key will highlight in glowing cyan.
   - Pinch thumb and index together: the key flashes green and types the character into whatever text field or document is active!

6. **Safety Master Switch**:
   - Toggle the top-right button `[ GESTURE CONTROL: ON / OFF ]` at any time to temporarily suspend all mouse, keyboard, and brightness dispatches while keeping camera tracking active.

7. **Customizing Settings**:
   - Click `[ SETTINGS ]` to adjust sensitivity sliders, mouse speed, smoothing factors, or camera device index. Click `SAVE SETTINGS` to persist changes.

---

## 🛠️ Troubleshooting & Notes

- **Webcam Access / Permission**: Ensure no other application (Zoom, Teams, Discord) is currently using your webcam. If you have multiple cameras, change the camera index in `[ SETTINGS ]` from `0` to `1` or `2`.
- **Display Brightness on External Monitors**: Some external desktop monitors do not support software brightness via DDC/CI or Windows WMI. GestoControl detects this automatically, operates in Virtual Brightness mode without crashing, and visualizes the percentage on screen.
- **Lighting**: For best MediaPipe tracking, ensure your hand is evenly illuminated and not silhouetted against a bright window or backlit background.

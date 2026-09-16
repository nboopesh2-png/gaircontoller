"""
GestoControl – Python Hand Gesture Laptop Controller
Application Main Entry Point.
Validates runtime dependencies, initializes Tkinter desktop window,
and launches the interactive paired-camera interface.
"""

import sys
import tkinter as tk
from tkinter import messagebox


def check_dependencies():
    """
    Checks that all required Python packages are installed.
    Provides clear error dialogue if any essential package is missing.
    """
    missing_packages = []

    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")

    try:
        import mediapipe
    except ImportError:
        missing_packages.append("mediapipe")

    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")

    try:
        import pyautogui
    except ImportError:
        missing_packages.append("pyautogui")

    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_packages.append("Pillow")

    if missing_packages:
        root_dummy = tk.Tk()
        root_dummy.withdraw()
        pkgs_str = "\n  • " + "\n  • ".join(missing_packages)
        msg = (
            f"Missing required Python dependencies:\n{pkgs_str}\n\n"
            f"Please run the following command to install all requirements:\n\n"
            f"    pip install -r requirements.txt\n"
        )
        messagebox.showerror("GestoControl - Missing Dependencies", msg)
        root_dummy.destroy()
        sys.exit(1)


def main():
    """Launches GestoControl Desktop Application."""
    check_dependencies()

    from ui import GestoControlUI

    root = tk.Tk()
    root.title("GestoControl – Hand Gesture Laptop Controller")

    # Center window on screen
    window_width = 1180
    window_height = 820
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_pos = max(0, int((screen_width - window_width) / 2))
    y_pos = max(0, int((screen_height - window_height) / 2))
    root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")

    app = GestoControlUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

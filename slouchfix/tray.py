"""System tray icon for background operation: SlouchFix has no main window
most of the time, matching the "non-intrusive" pitch — it lives in the tray
and only speaks up via toast notifications.
"""

from __future__ import annotations

import threading

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from .app import SlouchFixApp


def _build_icon_image(color: str = "#2E86AB") -> Image.Image:
    """Small generated icon (a rounded posture/spine glyph) -- no asset file needed."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, size - 4, size - 4), fill=color)
    draw.line((size * 0.5, size * 0.2, size * 0.5, size * 0.8), fill="white", width=6)
    draw.line((size * 0.5, size * 0.35, size * 0.35, size * 0.5), fill="white", width=6)
    draw.line((size * 0.5, size * 0.55, size * 0.65, size * 0.7), fill="white", width=6)
    return img


class SlouchFixTray:
    def __init__(self) -> None:
        self.app = SlouchFixApp(show_preview=False, notifications=True)
        self._worker: threading.Thread | None = None
        self.icon = Icon(
            "SlouchFix",
            _build_icon_image(),
            "SlouchFix",
            menu=Menu(
                MenuItem(lambda item: "Resume" if self.app.paused else "Pause", self._toggle_pause),
                MenuItem("Recalibrate", self._recalibrate),
                MenuItem("Quit", self._quit),
            ),
        )

    def _toggle_pause(self, icon, item) -> None:
        self.app.toggle_pause()

    def _recalibrate(self, icon, item) -> None:
        self.app.recalibrate()

    def _quit(self, icon, item) -> None:
        self.app.stop()
        icon.stop()

    def run(self) -> None:
        self._worker = threading.Thread(target=self.app.run, daemon=True)
        self._worker.start()
        self.icon.run()  # blocks; runs the tray event loop on the main thread


def main() -> None:
    SlouchFixTray().run()


if __name__ == "__main__":
    main()

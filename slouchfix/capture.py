"""Thin wrapper around cv2.VideoCapture for webcam frame acquisition."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


class WebcamCapture:
    """Context-managed webcam reader. Raises RuntimeError if the camera can't open."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "WebcamCapture":
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fall back to default backend if DirectShow isn't available.
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that a camera is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first or use a context manager.")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

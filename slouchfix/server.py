"""Local HTTP+WebSocket backend for the Flutter desktop app
(`slouchfix_app/`). Wraps the same `SlouchFixApp` core the CLI/tray/Tkinter
entry points use, so this is an additional front end, not a rewrite --
`scripts/run_app.py`, `run_tray.py`, and `run_dashboard.py` keep working
unchanged.

Bound to 127.0.0.1 only: nothing here should ever be reachable off this
machine, matching the "no video ever leaves the machine" privacy property
described in the README.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config
from .app import SlouchFixApp
from .history import HistoryLogger, compute_period_stats
from .settings import Settings

app = FastAPI(title="SlouchFix backend")

PREVIEW_FPS = 10


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.settings = Settings.load()
        self.history = HistoryLogger()
        self.slouch_app: SlouchFixApp | None = None
        self.app_thread: threading.Thread | None = None
        self.start_error: str | None = None

    def latest_frame(self):
        if self.slouch_app is not None:
            return self.slouch_app.latest_frame
        return None


state = ServerState()


# ----- status ---------------------------------------------------------


@app.get("/api/status")
def get_status() -> dict[str, Any]:
    a = state.slouch_app
    s = a.state if a is not None else None
    return {
        "running": a is not None,
        "paused": a.paused if a is not None else False,
        "label": s.label if s is not None else None,
        "confidence": s.confidence if s is not None else None,
        "distance_cm": s.distance_cm if s is not None else None,
        "duration_sec": s.duration_sec if s is not None else None,
        "session_seconds": a.session_seconds if a is not None else 0.0,
        "notifications_enabled": state.settings.notifications_enabled,
        "start_error": state.start_error,
    }


# ----- monitoring control ------------------------------------------------


@app.post("/api/start")
def start_monitoring() -> dict[str, Any]:
    with state.lock:
        if state.slouch_app is not None and state.app_thread is not None and state.app_thread.is_alive():
            return {"ok": True, "already_running": True}

        state.start_error = None
        settings = Settings.load()
        state.settings = settings

        def build_and_run() -> None:
            try:
                built = SlouchFixApp(
                    show_preview=False,
                    notifications=settings.notifications_enabled,
                    settings=settings,
                    history=state.history,
                )
            except Exception as exc:  # camera busy/missing, etc.
                state.start_error = str(exc)
                return
            state.slouch_app = built
            built.run()

        state.app_thread = threading.Thread(target=build_and_run, daemon=True)
        state.app_thread.start()
    return {"ok": True}


@app.post("/api/stop")
def stop_monitoring() -> dict[str, Any]:
    with state.lock:
        a = state.slouch_app
        thread = state.app_thread
    if a is not None:
        a.stop()
    if thread is not None:
        thread.join(timeout=3.0)
    with state.lock:
        state.slouch_app = None
        state.app_thread = None
    return {"ok": True}


@app.post("/api/pause")
def toggle_pause() -> dict[str, Any]:
    a = state.slouch_app
    if a is None:
        raise HTTPException(status_code=409, detail="Not running.")
    a.toggle_pause()
    return {"ok": True, "paused": a.paused}


class NotificationsBody(BaseModel):
    enabled: bool


@app.post("/api/notifications")
def set_notifications(body: NotificationsBody) -> dict[str, Any]:
    state.settings.notifications_enabled = body.enabled
    state.settings.save()
    a = state.slouch_app
    if a is not None:
        a.notifier.enabled = body.enabled
    return {"ok": True}


# ----- settings ---------------------------------------------------------


class SettingsBody(BaseModel):
    confirm_frames: int
    notification_cooldown_sec: float
    good_posture_reminder_sec: float
    notifications_enabled: bool
    camera_index: int


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return state.settings.__dict__.copy()


@app.put("/api/settings")
def put_settings(body: SettingsBody) -> dict[str, Any]:
    for key, value in body.model_dump().items():
        setattr(state.settings, key, value)
    state.settings.save()
    note = None
    if state.slouch_app is not None:
        note = "Restart monitoring (Stop then Start) to apply the new settings."
    return {"ok": True, "note": note}


# ----- history / stats ---------------------------------------------------


@app.get("/api/history")
def get_history(type: str | None = None, since: str | None = None) -> list[dict[str, Any]]:
    from datetime import datetime

    since_dt = datetime.fromisoformat(since) if since else None

    if type == "state":
        return [{"type": "state", **row} for row in state.history.load_states(since=since_dt)]
    if type == "alert":
        return [{"type": "alert", **row} for row in state.history.load_alerts(since=since_dt)]

    events = [{"type": "state", **row} for row in state.history.load_states(since=since_dt)]
    events += [{"type": "alert", **row} for row in state.history.load_alerts(since=since_dt)]
    events.sort(key=lambda r: r["timestamp"], reverse=True)
    return events[:300]


@app.get("/api/stats")
def get_stats(period: str = "today") -> dict[str, Any]:
    from datetime import datetime, timedelta

    since = None
    if period == "today":
        since = datetime.combine(datetime.now().date(), datetime.min.time())
    elif period == "week":
        since = datetime.now() - timedelta(days=7)

    stats = compute_period_stats(state.history, since=since)
    return {
        "tracked_minutes": stats.tracked_minutes,
        "label_seconds": stats.label_seconds,
        "alert_counts": dict(stats.alert_counts),
    }


# ----- live camera preview ------------------------------------------------


@app.websocket("/ws/preview")
async def ws_preview(websocket: WebSocket) -> None:
    await websocket.accept()
    interval = 1.0 / PREVIEW_FPS
    try:
        while True:
            frame = state.latest_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    await websocket.send_bytes(buf.tobytes())
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return


def create_app() -> FastAPI:
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8722, log_level="info")


if __name__ == "__main__":
    main()

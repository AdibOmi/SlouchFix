"""Tkinter dashboard: a single window for the things that were previously
tray-menu-only or invisible -- starting/stopping monitoring, pausing,
recalibrating, editing thresholds, and browsing history/stats. Runs the
monitoring loop in a background thread (same pattern as `tray.py`) while
Tkinter owns the main thread.
"""

from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import messagebox, ttk

from . import config
from .app import SlouchFixApp
from .history import HistoryLogger, compute_period_stats
from .settings import Settings

LABEL_TITLES = {
    "good_posture": "Good Posture",
    "slouched": "Slouched",
    "leaning_forward": "Leaning Forward",
    "too_close": "Too Close",
    "head_tilted": "Head Tilted",
    "looking_away": "Looking Away",
}

LABEL_COLORS = {
    "good_posture": "#2E8B57",
    "slouched": "#C0392B",
    "leaning_forward": "#D35400",
    "too_close": "#8E44AD",
    "head_tilted": "#B7950B",
    "looking_away": "#2980B9",
}

SETTINGS_FIELDS = [
    ("too_close_distance_cm", "Too-close distance (cm)", float),
    ("yaw_looking_away_deg", "Looking-away yaw threshold (deg)", float),
    ("roll_head_tilted_deg", "Head-tilt roll threshold (deg)", float),
    ("pitch_leaning_forward_deg", "Leaning-forward pitch threshold (deg)", float),
    ("notification_cooldown_sec", "Notification cooldown (sec)", float),
    ("good_posture_reminder_sec", "Good-posture reminder interval (sec)", float),
    ("confirm_frames", "Frames to confirm a state change", int),
    ("camera_index", "Camera index", int),
]


class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SlouchFix")
        self.geometry("640x560")
        self.minsize(560, 460)

        self.history = HistoryLogger()
        self.settings = Settings.load()
        self.app: SlouchFixApp | None = None
        self._app_thread: threading.Thread | None = None
        self._start_error: str | None = None
        self._stopping = False

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # ----- layout ---------------------------------------------------------

    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_tab = ttk.Frame(notebook)
        self.history_tab = ttk.Frame(notebook)
        self.stats_tab = ttk.Frame(notebook)
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.status_tab, text="Status")
        notebook.add(self.history_tab, text="History")
        notebook.add(self.stats_tab, text="Stats")
        notebook.add(self.settings_tab, text="Settings")

        self._build_status_tab()
        self._build_history_tab()
        self._build_stats_tab()
        self._build_settings_tab()

    def _build_status_tab(self) -> None:
        frame = self.status_tab

        self.label_var = tk.StringVar(value="Not running")
        self.detail_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")

        tk.Label(frame, textvariable=self.label_var, font=("Segoe UI", 20, "bold")).pack(pady=(20, 4))
        tk.Label(frame, textvariable=self.detail_var, font=("Segoe UI", 11)).pack()
        tk.Label(frame, textvariable=self.model_var, font=("Segoe UI", 9), fg="#666").pack(pady=(4, 20))

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=10)

        self.start_btn = ttk.Button(btn_row, text="Start Monitoring", command=self._start_monitoring)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.pause_btn = ttk.Button(btn_row, text="Pause", command=self._toggle_pause, state="disabled")
        self.pause_btn.grid(row=0, column=1, padx=5)
        self.recal_btn = ttk.Button(btn_row, text="Recalibrate", command=self._recalibrate, state="disabled")
        self.recal_btn.grid(row=0, column=2, padx=5)

        notif_row = ttk.Frame(frame)
        notif_row.pack(pady=(20, 0))
        self.notif_var = tk.BooleanVar(value=self.settings.notifications_enabled)
        ttk.Checkbutton(
            notif_row,
            text="Desktop notifications enabled",
            variable=self.notif_var,
            command=self._toggle_notifications,
        ).pack()

    def _build_history_tab(self) -> None:
        frame = self.history_tab
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(6, 0))
        ttk.Button(top, text="Refresh", command=self._refresh_history).pack(side="right", padx=6)

        columns = ("time", "type", "label", "details")
        self.history_tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
        for col, width in (("time", 140), ("type", 60), ("label", 130), ("details", 220)):
            self.history_tree.heading(col, text=col.capitalize())
            self.history_tree.column(col, width=width, anchor="w")
        self.history_tree.pack(fill="both", expand=True, padx=6, pady=6)
        self._refresh_history()

    def _build_stats_tab(self) -> None:
        frame = self.stats_tab
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(6, 0))

        self.period_var = tk.StringVar(value="today")
        for text, value in (("Today", "today"), ("Last 7 days", "week"), ("All time", "all")):
            ttk.Radiobutton(
                top, text=text, value=value, variable=self.period_var, command=self._refresh_stats
            ).pack(side="left", padx=6)
        ttk.Button(top, text="Refresh", command=self._refresh_stats).pack(side="right", padx=6)

        self.stats_summary_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.stats_summary_var, font=("Segoe UI", 10)).pack(
            anchor="w", padx=8, pady=(8, 4)
        )

        self.stats_canvas = tk.Canvas(frame, height=260, bg="white", highlightthickness=0)
        self.stats_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.stats_canvas.bind("<Configure>", lambda _e: self._refresh_stats())
        self._refresh_stats()

    def _build_settings_tab(self) -> None:
        frame = self.settings_tab
        form = ttk.Frame(frame)
        form.pack(fill="x", padx=12, pady=12)

        self.setting_vars: dict[str, tk.Variable] = {}
        for row, (key, label, kind) in enumerate(SETTINGS_FIELDS):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            var = tk.DoubleVar(value=getattr(self.settings, key)) if kind is float else tk.IntVar(
                value=getattr(self.settings, key)
            )
            self.setting_vars[key] = var
            ttk.Entry(form, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=8)

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=10)
        ttk.Button(btn_row, text="Save", command=self._save_settings).grid(row=0, column=0, padx=5)
        ttk.Button(btn_row, text="Reset to Defaults", command=self._reset_settings).grid(row=0, column=1, padx=5)

        self.settings_note_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.settings_note_var, foreground="#666").pack(pady=(0, 10))

    # ----- app control ------------------------------------------------------

    def _start_monitoring(self) -> None:
        if self._app_thread is not None and self._app_thread.is_alive():
            return
        self._start_error = None
        self._stopping = False
        settings = Settings.load()

        def build_and_run() -> None:
            try:
                self.app = SlouchFixApp(
                    show_preview=False,
                    notifications=settings.notifications_enabled,
                    settings=settings,
                    history=self.history,
                )
            except Exception as exc:  # camera busy/missing, no face detector assets, etc.
                self._start_error = str(exc)
                return
            self.app.run()

        self._app_thread = threading.Thread(target=build_and_run, daemon=True)
        self._app_thread.start()
        self.start_btn.configure(text="Stop Monitoring", command=self._stop_monitoring)
        self.pause_btn.configure(state="normal", text="Pause")
        self.recal_btn.configure(state="normal")

    def _stop_monitoring(self) -> None:
        self._stopping = True
        if self.app is not None:
            self.app.stop()
        self.start_btn.configure(text="Stopping...", state="disabled")
        self.pause_btn.configure(state="disabled")
        self.recal_btn.configure(state="disabled")
        self.after(300, self._finish_stop)

    def _finish_stop(self) -> None:
        if self._app_thread is not None and self._app_thread.is_alive():
            self.after(300, self._finish_stop)
            return
        self.app = None
        self._stopping = False
        self.label_var.set("Not running")
        self.detail_var.set("")
        self.model_var.set("")
        self.start_btn.configure(text="Start Monitoring", command=self._start_monitoring, state="normal")
        self.pause_btn.configure(state="disabled", text="Pause")
        self.recal_btn.configure(state="disabled")

    def _toggle_pause(self) -> None:
        if self.app is None:
            return
        self.app.toggle_pause()
        self.pause_btn.configure(text="Resume" if self.app.paused else "Pause")

    def _recalibrate(self) -> None:
        if self.app is None:
            return
        threading.Thread(target=self.app.recalibrate, daemon=True).start()

    def _toggle_notifications(self) -> None:
        self.settings.notifications_enabled = self.notif_var.get()
        self.settings.save()
        if self.app is not None:
            self.app.notifier.enabled = self.notif_var.get()

    # ----- polling ------------------------------------------------------

    def _tick(self) -> None:
        if self._start_error and not self._stopping:
            err = self._start_error
            self._start_error = None
            messagebox.showerror("SlouchFix", f"Could not start monitoring:\n{err}")
            self._finish_stop()
        elif self.app is not None:
            state = self.app.state
            if state is None:
                self.label_var.set("Looking for a face...")
                self.detail_var.set("")
            else:
                self.label_var.set(LABEL_TITLES.get(state.label, state.label))
                mins, secs = divmod(int(state.duration_sec), 60)
                detail = (
                    f"Confidence {state.confidence:.0%} | Distance {state.distance_cm:.0f} cm | "
                    f"{mins}m {secs}s in this state"
                )
                if self.app.paused:
                    detail += "  [PAUSED]"
                self.detail_var.set(detail)
            model_kind = "trained model" if self.app.using_trained_model else "rule-based baseline"
            session_min = int(self.app.session_seconds // 60)
            self.model_var.set(f"Using {model_kind} | session: {session_min} min")

        self.after(500, self._tick)

    # ----- history / stats ------------------------------------------------

    def _refresh_history(self) -> None:
        for row in self.history_tree.get_children():
            self.history_tree.delete(row)

        events = [{"type": "state", **row} for row in self.history.load_states()]
        events += [{"type": "alert", **row} for row in self.history.load_alerts()]
        events.sort(key=lambda r: r["timestamp"], reverse=True)

        for row in events[:300]:
            if row["type"] == "state":
                details = f"confidence {row['confidence']} | {row['distance_cm']} cm"
            else:
                details = row.get("message", "")
            self.history_tree.insert(
                "",
                "end",
                values=(row["timestamp"], row["type"], LABEL_TITLES.get(row["label"], row["label"]), details),
            )

    def _refresh_stats(self) -> None:
        period = self.period_var.get()
        since = None
        if period == "today":
            since = datetime.combine(datetime.now().date(), datetime.min.time())
        elif period == "week":
            since = datetime.now() - timedelta(days=7)

        stats = compute_period_stats(self.history, since=since)
        alert_total = sum(stats.alert_counts.values())
        self.stats_summary_var.set(f"Tracked: {stats.tracked_minutes:.0f} min  |  Alerts sent: {alert_total}")

        self.stats_canvas.delete("all")
        max_seconds = max(stats.label_seconds.values(), default=0.0) or 1.0
        canvas_width = self.stats_canvas.winfo_width() or 560
        bar_h = 28
        y = 10
        for label in config.LABELS:
            seconds = stats.label_seconds.get(label, 0.0)
            width = int((seconds / max_seconds) * (canvas_width - 220))
            self.stats_canvas.create_text(8, y + bar_h / 2, anchor="w", text=LABEL_TITLES.get(label, label))
            self.stats_canvas.create_rectangle(
                160, y, 160 + max(width, 2), y + bar_h - 6, fill=LABEL_COLORS.get(label, "#999"), width=0
            )
            alerts = stats.alert_counts.get(label, 0)
            self.stats_canvas.create_text(
                166 + max(width, 2), y + bar_h / 2, anchor="w", text=f"{seconds / 60.0:.0f} min, {alerts} alerts"
            )
            y += bar_h + 6

    # ----- settings ---------------------------------------------------------

    def _save_settings(self) -> None:
        try:
            for key, var in self.setting_vars.items():
                setattr(self.settings, key, var.get())
        except tk.TclError:
            messagebox.showerror("SlouchFix", "One of the settings fields has an invalid number.")
            return
        self.settings.notifications_enabled = self.notif_var.get()
        self.settings.save()
        note = "Saved."
        if self.app is not None:
            note += " Restart monitoring (Stop then Start) to apply the new thresholds."
        self.settings_note_var.set(note)

    def _reset_settings(self) -> None:
        self.settings = Settings()
        for key, var in self.setting_vars.items():
            var.set(getattr(self.settings, key))
        self.notif_var.set(self.settings.notifications_enabled)
        self.settings.save()
        self.settings_note_var.set("Reset to defaults and saved.")

    # ----- shutdown ---------------------------------------------------------

    def _on_close(self) -> None:
        if self.app is not None:
            self.app.stop()
            if self._app_thread is not None:
                self._app_thread.join(timeout=2.0)
        self.destroy()


def main() -> None:
    Dashboard().mainloop()


if __name__ == "__main__":
    main()

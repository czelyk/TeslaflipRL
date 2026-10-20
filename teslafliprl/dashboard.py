"""Live episode dashboard. Importing this module never opens a GUI.

Create a Dashboard only for visible runs. Headless training can import the
module without importing matplotlib (or selecting a graphical backend).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import time
from typing import Callable, Mapping


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _success(value: object) -> bool:
    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "success", "başarılı"}
    return bool(value)


_PHASES = {
    "approach": "Rampaya yaklaşma", "drive": "Rampaya yaklaşma",
    "ramp": "Rampayı tırmanma", "climb": "Rampayı tırmanma",
    "airborne": "Havada", "flight": "Havada", "launch": "Rampadan ayrılma",
    "landed": "Yerde", "landing": "İniş", "settling": "İnişi dengeleme",
    "done": "Bölüm tamamlandı", "ground": "Zeminde",
}
_OUTCOMES = {
    "success": "Başarılı iniş", "successful_landing": "Başarılı iniş",
    "timeout": "Süre doldu", "crash": "Çarpışma", "hard_landing": "Sert iniş",
    "flipped": "Ters iniş", "upside_down": "Ters iniş", "side_landing": "Yan iniş",
    "failure": "Başarısız", "out_of_bounds": "Alan dışına çıktı",
    "running": "Devam ediyor", "": "Devam ediyor",
    "inverted_or_side_landing": "Ters / yan iniş", "spinning_landing": "Dönerek iniş",
    "unstable_landing": "Kararsız iniş", "fell_from_ramp": "Rampadan erken düştü",
    "ground_before_takeoff": "Fırlamadan yere düştü",
}
_DISPLAY_NAMES = {
    "front_up": "Ön ↑", "front_down": "Ön ↓",
    "rear_up": "Arka ↑", "rear_down": "Arka ↓",
    "left_up": "Sol ↑", "left_down": "Sol ↓",
    "right_up": "Sağ ↑", "right_down": "Sağ ↓",
    "front_left": "Ön ←", "front_right": "Ön →",
    "rear_left": "Arka ←", "rear_right": "Arka →",
    "bottom_lift": "Alt ↑",
}


class Dashboard:
    """A separate, non-blocking matplotlib window for training or watching.

    ``update`` must keep being called while paused so window events are served.
    ``speed=0`` means unlimited; ``on_speed_change`` receives that same value.
    Every call to ``finish_episode`` records one episode, regardless of frame
    throttling. All completed episodes remain available in ``history``.
    """

    REFRESH_HZ = 15.0

    def __init__(
        self,
        thruster_names: list[str],
        initial_speed: float = 1.0,
        on_speed_change: Callable[[float], None] | None = None,
        title: str = "TeslaFlipRL · Canlı İzleme",
    ) -> None:
        # Keep this import local: headless runs do not need a display backend.
        import matplotlib
        # The native Bullet window and Tk's blitter can stall each other's
        # Windows message processing. Qt keeps the live panel responsive.
        if sys.platform == 'win32' and not os.environ.get('MPLBACKEND'):
            os.environ.setdefault('QT_API', 'pyside6')
            matplotlib.use('QtAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self._on_speed_change = on_speed_change
        self.is_open = True
        self.paused = False
        self.speed = max(0.0, _number(initial_speed, 1.0))
        self.thruster_names = list(thruster_names) or ["İtici"]
        self.history: list[dict] = []
        self._telemetry: dict = {}
        self._last_refresh = 0.0
        self._episode_ids: list[int] = []
        self._returns: list[float] = []
        self._success_values: list[bool] = []
        self._success_rolling: list[float] = []
        self._last_finished_episode: int | None = None
        self._speed_buttons: list[tuple[float, object]] = []
        self._colors = {
            "background": "#101820", "panel": "#182630", "text": "#ecf3f5",
            "muted": "#a7bac5", "grid": "#344754", "reward": "#54cfb5",
            "success": "#68aaff", "accent": "#ffc66b", "failure": "#ff8b86",
        }
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
            self.figure = plt.figure(figsize=(13, 8), facecolor=self._colors["background"])
        try:
            self.figure.canvas.manager.set_window_title(title)
        except AttributeError:
            pass
        self.figure.canvas.mpl_connect("close_event", self._handle_close)
        self.figure.text(.045, .958, title, color=self._colors["text"], fontsize=19, weight="bold")
        self._headline = self.figure.text(.045, .911, "Bölüm 1 · Hazırlanıyor", color=self._colors["accent"], fontsize=12)
        self._last_result = self.figure.text(.60, .911, "Henüz tamamlanan bölüm yok", color=self._colors["muted"], fontsize=10)
        self._metrics = []
        for xpos in (.045, .275, .505, .735):
            self._metrics.append(self.figure.text(xpos, .851, "", va="top", color=self._colors["text"], fontsize=11, linespacing=1.65))
        self.reward_axes = self.figure.add_axes((.065, .395, .415, .285))
        self.success_axes = self.figure.add_axes((.555, .395, .405, .285))
        self.thruster_axes = self.figure.add_axes((.065, .155, .895, .145))
        self._style_axes(self.reward_axes, "Bölüm ödülü", "Ödül")
        self._style_axes(self.success_axes, "Başarı oranı · son 20 bölüm", "%")
        self._style_axes(self.thruster_axes, "İtici seviyeleri", "%")
        self.reward_axes.set_xlabel("Bölüm", color=self._colors["muted"])
        self.success_axes.set_xlabel("Bölüm", color=self._colors["muted"])
        self.reward_line, = self.reward_axes.plot([], [], color=self._colors["reward"], lw=1.8, marker='o', markersize=3)
        self.current_reward, = self.reward_axes.plot([], [], "o", color=self._colors["accent"], markersize=5)
        self.success_line, = self.success_axes.plot([], [], color=self._colors["success"], lw=1.8, marker='o', markersize=3)
        self.success_axes.set_ylim(-3, 103)
        self.success_axes.set_yticks([0, 25, 50, 75, 100])
        self._bars = self.thruster_axes.bar(range(len(self.thruster_names)), [0] * len(self.thruster_names), color=self._colors["reward"], width=.55)
        display_names = [_DISPLAY_NAMES.get(name, name) for name in self.thruster_names]
        self.thruster_axes.set_xticks(range(len(self.thruster_names)), display_names)
        self.thruster_axes.set_ylim(0, 105)
        self.thruster_axes.set_yticks([0, 50, 100])
        self._bar_labels = [self.thruster_axes.text(i, 3, "0%", ha="center", va="bottom", color=self._colors["text"], fontsize=9) for i in range(len(self.thruster_names))]
        self.figure.text(.065, .080, "Görsel hız:", color=self._colors["muted"], fontsize=10)
        speed_choices = [(0.25, "¼×"), (0.5, "½×"), (1.0, "1×"), (2.0, "2×"), (4.0, "4×"), (8.0, "8×"), (0.0, "Sınırsız")]
        for idx, (value, label) in enumerate(speed_choices):
            ax = self.figure.add_axes((.16 + idx * .079, .056, .071, .042))
            button = Button(ax, label, color=self._colors["panel"], hovercolor="#34505e")
            button.label.set_color(self._colors["text"])
            button.on_clicked(lambda _event, requested=value: self._set_speed(requested))
            self._speed_buttons.append((value, button))
        pause_ax = self.figure.add_axes((.745, .056, .153, .042))
        self._pause_button = Button(pause_ax, "Duraklat", color=self._colors["panel"], hovercolor="#34505e")
        self._pause_button.label.set_color(self._colors["text"])
        self._pause_button.on_clicked(self._toggle_pause)
        self.figure.text(.065, .018, "Her bölüm kaydedilir. Sarı nokta: devam eden bölüm. Pencereyi kapatmak çalışmayı durdurur.", color=self._colors["muted"], fontsize=9)
        self._update_button_colors()
        self._render()
        if matplotlib.get_backend().lower() != 'agg':
            plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _style_axes(self, axes: object, title: str, ylabel: str) -> None:
        axes.set_facecolor(self._colors["panel"])
        axes.set_title(title, color=self._colors["text"], loc="left", pad=12, fontsize=11)
        axes.set_ylabel(ylabel, color=self._colors["muted"])
        axes.tick_params(colors=self._colors["muted"], labelsize=9)
        for spine in axes.spines.values():
            spine.set_color(self._colors["grid"])
        axes.grid(axis="y", color=self._colors["grid"], linewidth=.6, alpha=.6)
        axes.set_axisbelow(True)

    def _handle_close(self, _event: object = None) -> None:
        self.is_open = False
        self.paused = False

    def _set_speed(self, value: float) -> None:
        self.speed = value
        if self._on_speed_change is not None:
            self._on_speed_change(value)
        self._update_button_colors()
        self.figure.canvas.draw_idle()

    def _toggle_pause(self, _event: object = None) -> None:
        self.paused = not self.paused
        self._pause_button.label.set_text("Devam et" if self.paused else "Duraklat")
        self._pause_button.color = "#665028" if self.paused else self._colors["panel"]
        self._pause_button.ax.set_facecolor(self._pause_button.color)
        self.figure.canvas.draw_idle()

    def _update_button_colors(self) -> None:
        for value, button in self._speed_buttons:
            selected = math.isclose(value, self.speed)
            button.color = "#315b59" if selected else self._colors["panel"]
            button.ax.set_facecolor(button.color)

    def update(self, telemetry: Mapping | None = None) -> None:
        """Refresh telemetry and process GUI events at most 15 times/second."""
        if not self.is_open:
            return
        if telemetry is not None:
            self._telemetry = dict(telemetry)
        now = time.monotonic()
        if now - self._last_refresh < 1.0 / self.REFRESH_HZ:
            return
        self._last_refresh = now
        self._render()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def finish_episode(self, summary: Mapping | None = None) -> None:
        """Record an episode even if no frame was displayed during that run."""
        data = dict(self._telemetry)
        if summary is not None:
            data.update(summary)
        episode = int(_number(data.get("episode"), len(self.history) + 1))
        episode_return = _number(data.get("episode_return", data.get("return", data.get("reward", 0))))
        successful = _success(data.get("success", False))
        data.update(episode=episode, episode_return=episode_return, success=successful)
        if not data.get("outcome") or data.get("outcome") == "running":
            data["outcome"] = "success" if successful else "failure"
        self.history.append(dict(data))
        self._episode_ids.append(episode)
        self._returns.append(episode_return)
        self._success_values.append(successful)
        recent = self._success_values[-20:]
        self._success_rolling.append(100.0 * sum(recent) / len(recent))
        self._last_finished_episode = episode
        self._telemetry = data
        if self.is_open:
            # Keep event handling/redraw throttled even for very short episodes.
            self.update()

    def _render(self) -> None:
        data = self._telemetry
        episode = int(_number(data.get("episode"), len(self.history) + 1))
        current_return = _number(data.get("episode_return", data.get("return", 0)))
        phase = str(data.get("phase", "Hazırlanıyor"))
        outcome = str(data.get("outcome", ""))
        completed = episode == self._last_finished_episode
        description = _OUTCOMES.get(outcome, outcome) if completed else _PHASES.get(phase, phase)
        self._headline.set_text(f"Bölüm {episode}  ·  {description}")
        self._headline.set_color(self._colors["reward"] if completed and _success(data.get("success")) else self._colors["failure"] if completed else self._colors["accent"])
        if self.history:
            last = self.history[-1]
            last_outcome = str(last.get("outcome", ""))
            last_description = _OUTCOMES.get(last_outcome, last_outcome)
            self._last_result.set_text(f"Son sonuç · Bölüm {last['episode']}: {last_description}")
            self._last_result.set_color(self._colors["reward"] if last["success"] else self._colors["failure"])
        contacts = data.get("wheel_contacts", 0)
        if hasattr(contacts, "__iter__") and not isinstance(contacts, (str, Mapping)):
            contacts = sum(bool(value) for value in contacts)
        self._metrics[0].set_text(f"Kümülatif ödül  {current_return:,.2f}\nAdım ödülü  {_number(data.get('reward')):+.3f}")
        self._metrics[1].set_text(f"Hız  {_number(data.get('speed_m_s')):.2f} m/s\nDüşey hız  {_number(data.get('vertical_speed_m_s')):+.2f} m/s")
        self._metrics[2].set_text(f"Roll / pitch / yaw (°)\n{_number(data.get('roll_deg')):+.1f} / {_number(data.get('pitch_deg')):+.1f} / {_number(data.get('yaw_deg')):+.1f}")
        self._metrics[3].set_text(f"Açısal hız  {_number(data.get('angular_speed_rad_s')):.2f} rad/s\nSüre  {_number(data.get('elapsed')):.1f} s  ·  Teker  {int(_number(contacts))}/4")
        total_force = sum(_number(x) for x in data.get('thruster_forces_n', []))
        self.thruster_axes.set_title(f"İtici seviyeleri · {data.get('thruster_status', '')} · {total_force:.0f} N · Biriken itki {_number(data.get('total_thruster_impulse_ns')):.0f} N·s",
                                     color=self._colors['text'], loc='left', pad=12, fontsize=10)
        self.reward_line.set_data(self._episode_ids, self._returns)
        self.current_reward.set_data([] if completed else [episode], [] if completed else [current_return])
        self.success_line.set_data(self._episode_ids, self._success_rolling)
        max_episode = max([episode, 5] + self._episode_ids)
        min_episode = min([1, episode] + self._episode_ids)
        self.reward_axes.set_xlim(min_episode - .5, max_episode + .5)
        self.success_axes.set_xlim(min_episode - .5, max_episode + .5)
        values = self._returns + [current_return, 0]
        low, high = min(values), max(values)
        margin = max(2.0, (high - low) * .12)
        self.reward_axes.set_ylim(low - margin, high + margin)
        levels = data.get("thrusters", [])
        if isinstance(levels, Mapping):
            levels = [levels.get(name, 0) for name in self.thruster_names]
        if levels is None:
            levels = []
        try:
            levels = list(levels)
        except TypeError:
            levels = []
        for idx, (bar, label) in enumerate(zip(self._bars, self._bar_labels)):
            level = min(1.0, max(0.0, _number(levels[idx] if idx < len(levels) else 0)))
            percent = level * 100.0
            bar.set_height(percent)
            label.set_text(f"{percent:.0f}%")
            label.set_y(min(98, percent + 3))

    def save(self, path: str | Path) -> Path:
        """Save a full-resolution dashboard screenshot without opening a dialog."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._render()
        self.figure.savefig(destination, dpi=140, facecolor=self.figure.get_facecolor())
        return destination

    def close(self) -> None:
        if self.is_open:
            self._plt.close(self.figure)
        self.is_open = False
        self.paused = False

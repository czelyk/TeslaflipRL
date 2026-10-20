"""Keep the native Bullet message loop isolated from the chart window's GUI loop."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time
import traceback


def _panel_worker(inbox, outbox, names, speed):
    from .dashboard import Dashboard
    dashboard = None
    try:
        dashboard = Dashboard(names, initial_speed=speed,
                              on_speed_change=lambda value: outbox.put(('speed', value)))
        outbox.put(('ready', None))
        previous_pause = False
        running = True
        while running and dashboard.is_open:
            latest = None
            for _ in range(256):
                try:
                    kind, data = inbox.get_nowait()
                except Empty:
                    break
                if kind == 'telemetry':
                    latest = data
                elif kind == 'episode':
                    dashboard.finish_episode(data)
                elif kind == 'save':
                    dashboard.save(data)
                elif kind == 'close':
                    running = False
                    break
            dashboard.update(latest)
            if dashboard.paused != previous_pause:
                previous_pause = dashboard.paused
                outbox.put(('paused', previous_pause))
            time.sleep(.012)
    except Exception:
        outbox.put(('error', traceback.format_exc()))
    finally:
        if dashboard is not None:
            dashboard.close()
        outbox.put(('closed', None))


class LivePanel:
    """Small, nonblocking parent facade matching Dashboard's public interface."""
    def __init__(self, names, initial_speed=1., on_speed_change=None):
        context = mp.get_context('spawn')
        self._inbox = context.Queue(maxsize=128)
        self._outbox = context.Queue()
        self._on_speed_change = on_speed_change
        self.is_open = True
        self.paused = False
        self._last_send = 0.
        self._process = context.Process(target=_panel_worker,
                    args=(self._inbox, self._outbox, names, initial_speed), daemon=True)
        self._process.start()
        try:
            kind, data = self._outbox.get(timeout=20)
            if kind != 'ready':
                raise RuntimeError(f'Canlı panel açılamadı: {data}')
        except Exception:
            self.close()
            raise

    def _receive(self):
        while True:
            try:
                kind, data = self._outbox.get_nowait()
            except Empty:
                break
            if kind == 'speed' and self._on_speed_change:
                self._on_speed_change(data)
            elif kind == 'paused':
                self.paused = data
            elif kind in ('closed', 'error'):
                self.is_open = False
                self.paused = False
                if kind == 'error':
                    print(f'Canlı panel hatası:\n{data}', flush=True)
        if not self._process.is_alive():
            self.is_open = False
            self.paused = False

    def update(self, telemetry=None):
        self._receive()
        now = time.monotonic()
        if self.is_open and telemetry is not None and now - self._last_send >= 1 / 15:
            try:
                self._inbox.put_nowait(('telemetry', dict(telemetry)))
                self._last_send = now
            except Full:
                pass  # Only a display frame is skipped; episode records are reliable.

    def finish_episode(self, summary):
        self._receive()
        if self.is_open:
            self._inbox.put(('episode', dict(summary)), timeout=5)

    def save(self, path):
        path = Path(path)
        if self.is_open:
            self._inbox.put(('save', str(path)), timeout=5)
        return path

    def close(self):
        if self._process.is_alive():
            try:
                self._inbox.put(('close', None), timeout=2)
            except Full:
                pass
            self._process.join(timeout=8)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        self._inbox.cancel_join_thread()
        self._inbox.close()
        self._outbox.close()
        self.is_open = False
        self.paused = False

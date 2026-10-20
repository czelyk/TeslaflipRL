"""Every real episode is displayed and journalled, including during PPO collection."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import gymnasium as gym


class EpisodeMonitor(gym.Wrapper):
    def __init__(self, env, run_dir, dashboard=False, speed=1.0):
        super().__init__(env)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_enabled = dashboard
        self.dashboard = None
        self.initial_speed = speed
        self.completed = 0
        self.successes = 0
        self.episode_return = 0.0
        self.episode_steps = 0
        self.viewer_closed = False
        self.history = []
        self.journal = self.run_dir / 'episodes.jsonl'
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.history.append(row)
            self.completed = len(self.history)
            self.successes = sum(bool(r.get('success')) for r in self.history)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_steps = 0
        if self.dashboard_enabled and self.dashboard is None:
            from .live_panel import LivePanel
            self.dashboard = LivePanel(
                list(self.unwrapped.telemetry['thruster_names']),
                initial_speed=self.initial_speed,
                on_speed_change=self.unwrapped.set_speed,
            )
            for row in self.history:
                self.dashboard.finish_episode(row)
        self._update()
        return obs, info

    def _telemetry(self):
        data = dict(self.unwrapped.telemetry)
        data.update(episode=self.completed + 1, episode_return=self.episode_return)
        return data

    def _update(self):
        self.viewer_closed = self.viewer_closed or bool(getattr(self.unwrapped, 'viewer_closed', False))
        if self.dashboard:
            self.dashboard.update(self._telemetry())
            self.viewer_closed = self.viewer_closed or not self.dashboard.is_open

    def step(self, action):
        if self.dashboard:
            while self.dashboard.is_open and self.dashboard.paused:
                self._update()
                time.sleep(0.02)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_steps += 1
        self._update()
        if terminated or truncated:
            self.completed += 1
            success = bool(info.get('is_success', False))
            self.successes += int(success)
            row = self._telemetry()
            row.update(episode=self.completed, success=success,
                       outcome=info.get('outcome', 'timeout' if truncated else 'ended'),
                       steps=self.episode_steps, impact_speed=info.get('impact_speed', 0.0),
                       success_rate=self.successes / self.completed,
                       terminated=bool(terminated), truncated=bool(truncated))
            self.history.append(row)
            with self.journal.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            csv_path = self.run_dir / 'episodes.csv'
            fields = ['episode', 'episode_return', 'success', 'outcome', 'steps', 'impact_speed', 'success_rate']
            exists = csv_path.exists()
            with csv_path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            info['episode'] = {'r': self.episode_return, 'l': self.episode_steps}
            print(f"Episode {self.completed}: {row['outcome']} | reward={self.episode_return:.2f} | success={self.successes}/{self.completed} | thrust={row.get('total_thruster_impulse_ns', 0):.1f} N.s", flush=True)
            if self.dashboard:
                self.dashboard.finish_episode(row)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.dashboard:
            if self.dashboard.is_open:
                self.dashboard.save(self.run_dir / 'dashboard.png')
            self.dashboard.close()
        self.env.close()

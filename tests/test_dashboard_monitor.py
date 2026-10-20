"""Episode accounting and controls stay correct even when frames are throttled."""
import json
import time

import matplotlib
import numpy as np

from teslafliprl import EnvConfig, TeslaFlipEnv
from teslafliprl.monitor import EpisodeMonitor


def test_dashboard_preserves_every_episode_and_speed_pause_controls(tmp_path, monkeypatch):
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    matplotlib.use('Agg', force=True)
    from teslafliprl.dashboard import Dashboard
    requests = []
    dashboard = Dashboard(['front_up', 'bottom_lift'], on_speed_change=requests.append)
    try:
        # Simulate many completed episodes inside a single visual refresh period.
        dashboard._last_refresh = time.monotonic() + 10
        for episode in range(1, 26):
            dashboard.finish_episode({'episode': episode, 'episode_return': -episode,
                                      'success': episode % 2 == 0, 'outcome': 'success' if episode % 2 == 0 else 'hard_landing'})
        assert len(dashboard.history) == 25
        assert dashboard._success_rolling[-1] == 50
        dashboard._set_speed(0)
        dashboard._set_speed(.25)
        assert requests == [0, .25]
        dashboard._toggle_pause()
        assert dashboard.paused
        dashboard._toggle_pause()
        assert not dashboard.paused
        assert dashboard.save(tmp_path / 'panel.png').stat().st_size > 1000
    finally:
        dashboard.close()
    assert not dashboard.is_open


def test_monitor_appends_completed_episodes_and_keeps_exact_return(tmp_path):
    for _ in range(2):
        env = EpisodeMonitor(TeslaFlipEnv(config=EnvConfig(randomize=False)), tmp_path)
        total = 0
        try:
            env.reset(seed=42)
            for _ in range(1000):
                _, reward, terminated, truncated, _ = env.step(np.full(env.action_space.shape, -1, dtype=np.float32))
                total += reward
                if terminated or truncated:
                    break
            else:
                raise AssertionError('Episode did not complete')
            assert env.history[-1]['episode_return'] == total
        finally:
            env.close()
    rows = [json.loads(line) for line in (tmp_path / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['episode'] for row in rows] == [1, 2]
    assert len((tmp_path / 'episodes.csv').read_text(encoding='utf-8').splitlines()) == 3


def test_closed_native_viewer_stops_without_autoreset_error():
    import pybullet as p
    env = TeslaFlipEnv()
    try:
        initial, _ = env.reset(seed=42)
        p.disconnect(physicsClientId=env.client_id)
        obs, reward, terminated, truncated, info = env.step(np.full(env.action_space.shape, -1, dtype=np.float32))
        assert env.viewer_closed and info['viewer_closed']
        assert not terminated and not truncated and reward == 0
        np.testing.assert_array_equal(obs, initial)
    finally:
        env.close()

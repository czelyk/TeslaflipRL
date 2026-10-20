"""Every completed episode must survive in the on-disk journal."""
import csv
import json

import numpy as np

from teslafliprl import EnvConfig, TeslaFlipEnv
from teslafliprl.monitor import EpisodeMonitor


def test_each_episode_is_journalled_and_recovered(tmp_path):
    monitor = EpisodeMonitor(
        TeslaFlipEnv(config=EnvConfig(randomize=False, max_episode_seconds=1 / 60)),
        tmp_path,
    )
    try:
        for seed in (5, 6):
            monitor.reset(seed=seed)
            _, reward, terminated, truncated, info = monitor.step(
                np.full(monitor.action_space.shape, -1, dtype=np.float32)
            )
            assert not terminated and truncated
            assert info["episode"]["r"] == reward
        assert monitor.completed == 2
    finally:
        monitor.close()

    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["episode"] for row in rows] == [1, 2]
    assert all(row["outcome"] == "timeout" and not row["success"] for row in rows)
    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert [float(row["episode_return"]) for row in csv_rows] == [row["episode_return"] for row in rows]

    recovered = EpisodeMonitor(TeslaFlipEnv(), tmp_path)
    try:
        assert recovered.completed == 2
        assert len(recovered.history) == 2
    finally:
        recovered.close()

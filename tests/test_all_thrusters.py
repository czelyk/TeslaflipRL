"""Each actual car nozzle must produce measured impulses and moments."""
import numpy as np
import json
import pybullet as p
import pytest

from teslafliprl import EnvConfig, TeslaFlipEnv
from teslafliprl.thruster_diagnostics import pulse_response
from teslafliprl.vehicle import THRUSTERS, nozzle_position


@pytest.mark.parametrize('index', range(len(THRUSTERS)), ids=[t.name for t in THRUSTERS])
@pytest.mark.parametrize('orientation', [(0, 0, 0), (.3, .7, -.5)])
def test_every_actual_car_thruster_impulse_and_torque(index, orientation):
    row = pulse_response(index, orientation=orientation)
    assert row['passed'], row


def test_external_nozzle_outlets_preserve_actual_force_moments():
    for thruster in THRUSTERS:
        np.testing.assert_allclose(np.cross(nozzle_position(thruster), thruster.direction),
                                   np.cross(thruster.position, thruster.direction), atol=1e-12)


def test_real_ramp_flight_applies_and_renders_thrust():
    histories = []
    for fire in [False, True]:
        env = TeslaFlipEnv(config=EnvConfig(randomize=False), render_mode='rgb_array')
        try:
            env.reset(seed=42)
            action = np.full(env.action_space.shape, -1., dtype=np.float32)
            flight_steps = 0
            for _ in range(600):
                if env.phase == 'airborne':
                    flight_steps += 1
                    if fire:
                        action[0] = 1.
                _, _, done, timeout, _ = env.step(action)
                if flight_steps == 12:
                    _, omega = p.getBaseVelocity(env.car_id, physicsClientId=env.client_id)
                    histories.append(np.asarray(omega))
                    if fire:
                        assert env.telemetry['thruster_forces_n'][0] == 2000
                        assert env.telemetry['active_thrusters'] == ['front_up']
                        assert env.telemetry['total_thruster_impulse_ns'] > 0
                        env.render()
                        shapes = {row[1]: row for row in p.getVisualShapeData(env.car_id, physicsClientId=env.client_id)}
                        assert shapes[4 + len(THRUSTERS)][7][3] == 1
                        assert len(shapes) == 1 + 4 + 2 * len(THRUSTERS)
                    break
                assert not (done or timeout)
            else:
                pytest.fail('No airborne thrust window')
        finally:
            env.close()
    assert np.linalg.norm(histories[1] - histories[0]) > .2


def test_watch_cli_fires_each_of_13_channels_during_real_ramp_episodes(tmp_path):
    from watch import main
    main(['--headless', '--controller', 'thruster-test', '--episodes', '13', '--run-dir', str(tmp_path)])
    rows = [json.loads(row) for row in (tmp_path / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 13
    for index, row in enumerate(rows):
        assert row['total_thruster_impulse_ns'] > 0
        peaks = row['thruster_peak_forces_n']
        assert peaks[index] == 2000
        assert sum(peak > 0 for peak in peaks) == 1
        assert row['thruster_impulses_ns'][index] > 0

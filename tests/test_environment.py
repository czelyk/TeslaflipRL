"""Environment contract, complete physical launch, and controlled impact cases."""
import math

from gymnasium.utils.env_checker import check_env
import numpy as np
import pybullet as p
import pytest

from teslafliprl import EnvConfig, TeslaFlipEnv


@pytest.fixture
def env():
    instance = TeslaFlipEnv(config=EnvConfig(randomize=False))
    try:
        yield instance
    finally:
        instance.close()


def off_action(env):
    return np.full(env.action_space.shape, -1, dtype=np.float32)


def test_gymnasium_contract(env):
    check_env(env, skip_render_check=True)


def test_same_seed_reproduces_initial_state_and_trajectory():
    instance = TeslaFlipEnv()
    try:
        histories = []
        for _ in range(2):
            obs, _ = instance.reset(seed=736)
            history = [obs.copy()]
            for _ in range(50):
                obs, reward, terminated, truncated, _ = instance.step(off_action(instance))
                history.append(np.concatenate([obs, [reward, terminated, truncated]]))
                if terminated or truncated:
                    break
            histories.append(history)
        assert len(histories[0]) == len(histories[1])
        for first, second in zip(*histories):
            np.testing.assert_allclose(first, second, atol=1e-7, rtol=0)
    finally:
        instance.close()


@pytest.mark.parametrize("bad_action", [
    np.zeros(12), np.zeros((13, 1)), np.full(13, float("nan")),
    np.full(13, float("inf")), np.full(13, 2.0),
])
def test_invalid_actions_fail_before_advancing_physics(env, bad_action):
    env.reset(seed=4)
    before = p.getBasePositionAndOrientation(env.car_id, physicsClientId=env.client_id)
    with pytest.raises(ValueError):
        env.step(bad_action)
    after = p.getBasePositionAndOrientation(env.car_id, physicsClientId=env.client_id)
    assert env.elapsed == 0
    assert before == after


def test_ramp_approach_disables_thrusters(env):
    env.reset(seed=8)
    _, _, _, _, info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    assert info["phase"] == "approach"
    np.testing.assert_array_equal(env.telemetry["thrusters"], np.zeros(13))
    assert info["reward_terms"].get("fuel", 0) == 0


def test_complete_no_thrust_launch_is_contact_driven_and_leftward(env, monkeypatch):
    env.reset(seed=42)
    start, _ = p.getBasePositionAndOrientation(env.car_id, physicsClientId=env.client_id)
    velocity, _ = p.getBaseVelocity(env.car_id, physicsClientId=env.client_id)
    assert start[0] < 0 and velocity[0] > 0

    def teleport_forbidden(*args, **kwargs):
        raise AssertionError("step() must never reset a vehicle pose or velocity")

    for name in ("resetBasePositionAndOrientation", "resetBaseVelocity", "resetJointState"):
        monkeypatch.setattr(p, name, teleport_forbidden)

    previous = np.asarray(start)
    saw_ramp_contact = False
    maximum_height = start[2]
    for _ in range(int(7 * env.config.control_hz)):
        obs, reward, terminated, truncated, info = env.step(off_action(env))
        position, _ = p.getBasePositionAndOrientation(env.car_id, physicsClientId=env.client_id)
        position = np.asarray(position)
        assert np.linalg.norm(position - previous) < 2.0, "Discontinuous vehicle movement"
        assert np.all(np.isfinite(obs)) and math.isfinite(reward)
        previous = position
        maximum_height = max(maximum_height, position[2])
        contacts = p.getContactPoints(bodyA=env.car_id, physicsClientId=env.client_id)
        saw_ramp_contact |= any(c[2] in env.ramp_ids and c[9] > 0 for c in contacts)
        if info["takeoff_velocity"] is not None:
            assert saw_ramp_contact
            assert maximum_height > env.config.ramp_radius * 1.4
            assert info["takeoff_velocity"][0] < 0, "Ramp exit must launch toward the left"
            assert info["takeoff_velocity"][2] > 0
            return
        assert not (terminated or truncated), f"No physical takeoff: {info['outcome']}"
    pytest.fail("No physical takeoff within seven simulated seconds")


def prepare_landing(env, *, orientation=(0, 0, 0), height=0.72, vertical_speed=0):
    """Set a synthetic test case; production episodes always start at the ramp approach."""
    env.reset(seed=10)
    p.resetBasePositionAndOrientation(env.car_id, [-25, 0, height], p.getQuaternionFromEuler(orientation), physicsClientId=env.client_id)
    p.resetBaseVelocity(env.car_id, [0, 0, vertical_speed], [0, 0, 0], physicsClientId=env.client_id)
    for wheel in env.wheel_links:
        p.resetJointState(env.car_id, wheel, 0, 0, physicsClientId=env.client_id)
    env.phase = "airborne"
    env._took_off = True
    env._airborne_time = 0.1


def finish_landing(env):
    for _ in range(240):
        _, reward, terminated, truncated, info = env.step(off_action(env))
        if terminated or truncated:
            return reward, info
    pytest.fail("Controlled landing test never terminated")


def test_low_speed_four_wheel_landing_succeeds(env):
    prepare_landing(env)
    reward, info = finish_landing(env)
    assert info["is_success"] and info["outcome"] == "success"
    assert info["wheel_contacts"] == 4
    assert info["impact_speed"] < env.config.max_landing_speed
    assert info["reward_terms"]["terminal"] > 0
    assert reward > 0


def test_high_downward_speed_landing_fails_with_penalty(env):
    prepare_landing(env, height=2.0, vertical_speed=-8)
    reward, info = finish_landing(env)
    assert not info["is_success"] and info["outcome"] == "hard_landing"
    assert info["impact_speed"] > env.config.max_landing_speed
    assert info["reward_terms"]["terminal"] < 0 and reward < 0


@pytest.mark.parametrize("roll", [math.pi, math.pi / 2])
def test_inverted_and_side_landings_fail_with_penalty(env, roll):
    prepare_landing(env, orientation=(roll, 0, 0), height=0.85, vertical_speed=-0.2)
    reward, info = finish_landing(env)
    assert not info["is_success"]
    assert info["outcome"] == "inverted_or_side_landing"
    assert info["reward_terms"]["terminal"] < 0 and reward < 0


def test_airborne_thrust_consumes_fuel(env):
    prepare_landing(env, height=20)
    _, _, _, _, off_info = env.step(off_action(env))
    assert off_info["reward_terms"].get("fuel", 0) == 0
    _, _, _, _, on_info = env.step(np.ones(env.action_space.shape, dtype=np.float32))
    assert on_info["reward_terms"]["fuel"] < 0
    np.testing.assert_array_equal(env.telemetry["thrusters"], np.ones(13))


def test_episode_timeout_is_truncation(env):
    env.reset(seed=4)
    env.config.max_episode_seconds = env.dt
    _, reward, terminated, truncated, info = env.step(off_action(env))
    assert truncated and not terminated
    assert info["outcome"] == "timeout" and not info["is_success"]
    assert reward < 0
    with pytest.raises(RuntimeError):
        env.step(off_action(env))


def test_rgb_render_returns_an_image():
    instance = TeslaFlipEnv(render_mode="rgb_array", config=EnvConfig(randomize=False))
    try:
        instance.reset(seed=42)
        image = instance.render()
        assert image.ndim == 3 and image.shape[2] == 3
        assert image.dtype == np.uint8 and image.max() > image.min()
    finally:
        instance.close()

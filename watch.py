"""Watch saved PPO or diagnostic controllers, using the exact training environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pybullet as p

from teslafliprl.env import TeslaFlipEnv
from teslafliprl.experiment import load_config, save_config
from teslafliprl.monitor import EpisodeMonitor


def stabilize(env):
    """Bounded greedy PD allocation for diagnostics, not a trained policy."""
    base = env.unwrapped
    action = np.full(base.action_space.shape, -1., dtype=np.float32)
    if base.telemetry.get('phase') not in ('airborne', 'landing'):
        return action
    _, quat = p.getBasePositionAndOrientation(base.car_id, physicsClientId=base.client_id)
    _, omega = p.getBaseVelocity(base.car_id, physicsClientId=base.client_id)
    rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    # Cross-product attitude error controls up alignment without forcing heading.
    error_world = np.cross(rotation[:, 2], np.array([0., 0., 1.]))
    desired = rotation.T @ (5500. * error_world - 2400. * np.asarray(omega))
    columns = []
    for thruster in base.thrusters:
        point = np.asarray(thruster.position)
        force = np.asarray(thruster.direction) * base.config.thruster_force
        columns.append(np.cross(point, force))
    allocation = np.asarray(columns).T
    levels = np.zeros(len(columns))
    scale = max(float(np.linalg.norm(allocation, 2) ** 2), 1.)
    for _ in range(30):
        levels = np.clip(levels - (allocation.T @ (allocation @ levels - desired)) / scale, 0., 1.)
    return (2 * levels - 1).astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description='PPO modelini veya tanı denetleyicisini izle.')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--controller', choices=['random', 'off', 'stabilize', 'thruster-test'], default='stabilize')
    parser.add_argument('--camera', choices=['overview', 'follow'], default='follow')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/watch'))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.speed < 0:
        parser.error('episodes >= 1, speed >= 0 olmalı.')
    config = load_config(args.model)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, args.run_dir)
    env = EpisodeMonitor(TeslaFlipEnv(config=config, render_mode=None if args.headless else 'human',
                                    speed=args.speed), args.run_dir,
                         dashboard=not args.headless and not args.no_dashboard, speed=args.speed)
    model = None
    env.unwrapped.camera_mode = args.camera
    if args.model:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        model = PPO.load(str(args.model), device='cpu')
    try:
        for i in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + i)
            env.action_space.seed(args.seed + i)
            done = False
            pulse_time = 0.
            while not done and not env.viewer_closed:
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                elif args.controller == 'random':
                    action = env.action_space.sample()
                elif args.controller == 'stabilize':
                    action = stabilize(env)
                elif args.controller == 'thruster-test':
                    action = np.full(env.action_space.shape, -1., dtype=np.float32)
                    if env.unwrapped.phase == 'airborne' and pulse_time < .4:
                        action[i % len(action)] = 1.
                        pulse_time += env.unwrapped.dt
                else:
                    action = np.full(env.action_space.shape, -1., dtype=np.float32)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if env.viewer_closed:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == '__main__':
    main()

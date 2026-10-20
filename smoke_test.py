"""End-to-end diagnostic: physical takeoff, PPO update, save/load, render and journal."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from gymnasium.utils.env_checker import check_env

from teslafliprl.config import EnvConfig
from teslafliprl.env import TeslaFlipEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Ayrıca yerel 3B ve canlı panelde tam episode çalıştır.')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/smoke'))
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    env = TeslaFlipEnv(config=EnvConfig(randomize=False), render_mode='rgb_array')
    report = {}
    try:
        check_env(env, skip_render_check=True)
        report['gymnasium_api'] = 'passed'
        obs, _ = env.reset(seed=42)
        phases = set()
        launch = None
        for _ in range(1200):
            obs, reward, terminated, truncated, info = env.step(np.full(env.action_space.shape, -1., dtype=np.float32))
            phases.add(info['phase'])
            if info.get('takeoff_velocity') and launch is None:
                launch = info['takeoff_velocity']
                Image.fromarray(env.render()).save(args.run_dir / 'takeoff.png')
            if terminated or truncated:
                break
        assert launch is not None and launch[0] < 0, f'Sola fiziksel kalkış yok: {info}'
        assert {'approach', 'ramp', 'airborne'} <= phases, phases
        report.update(physical_takeoff='passed', takeoff_velocity=launch, baseline_outcome=info['outcome'])
    finally:
        env.close()
    command = [sys.executable, 'train.py', '--headless', '--steps', str(args.steps),
               '--n-steps', '128', '--batch-size', '64', '--checkpoint-freq', '128',
               '--run-dir', str(args.run_dir / 'ppo')]
    subprocess.run(command, check=True)
    from stable_baselines3 import PPO
    model = PPO.load(str(args.run_dir / 'ppo/final_model.zip'), device='cpu')
    env = TeslaFlipEnv(config=EnvConfig(randomize=False))
    try:
        obs, _ = env.reset(seed=42)
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
    finally:
        env.close()
    report['ppo_train_save_load'] = 'passed'
    subprocess.run([sys.executable, 'watch.py', '--headless', '--model', str(args.run_dir / 'ppo/final_model.zip'),
                    '--episodes', '1', '--run-dir', str(args.run_dir / 'watch')], check=True)
    report['full_episode_model_watch'] = 'passed'
    if args.gui:
        subprocess.run([sys.executable, 'watch.py', '--episodes', '1', '--speed', '4',
                        '--run-dir', str(args.run_dir / 'gui')], check=True)
        report['native_gui_and_dashboard'] = 'passed'
    (args.run_dir / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""PPO training: every episode rendered by default; --headless for throughput."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from teslafliprl.env import TeslaFlipEnv
from teslafliprl.experiment import load_config, save_config
from teslafliprl.monitor import EpisodeMonitor


class ViewerCallback(BaseCallback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def _on_step(self):
        return not self.monitor.viewer_closed


def main(argv=None):
    parser = argparse.ArgumentParser(description='TeslaFlipRL PPO eğitimi; varsayılan her episode canlı 3B.')
    parser.add_argument('--steps', type=int, default=200_000, help='Ek kontrol adımı; PPO rollout uzunluğuna yuvarlanır.')
    parser.add_argument('--headless', action='store_true', help='3B pencere ve panel açmadan hızlı eğitim.')
    parser.add_argument('--no-dashboard', action='store_true')
    parser.add_argument('--speed', type=float, default=1.0, help='0=sınırsız, 0.5=yavaş, 4=hızlı; fizik dt değişmez.')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-freq', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.n_steps < 2 or args.batch_size < 2 or args.checkpoint_freq < 1 or args.speed < 0:
        parser.error('Adım/batch/checkpoint pozitif; n-steps ve batch en az 2, hız sıfır veya pozitif olmalı.')
    if not args.resume and args.n_steps % args.batch_size:
        parser.error('n-steps, batch-size değerinin katı olmalı.')
    torch.set_num_threads(1)
    run_dir = args.run_dir or Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.resume)
    save_config(config, run_dir)
    manifest = {'seed': args.seed, 'resume': str(args.resume) if args.resume else None,
                'headless': args.headless, 'versions': {p: version(p) for p in ['gymnasium', 'stable-baselines3', 'torch', 'numpy']}}
    (run_dir / 'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    env = EpisodeMonitor(TeslaFlipEnv(config=config, render_mode=None if args.headless else 'human',
                                    speed=args.speed), run_dir,
                         dashboard=not args.headless and not args.no_dashboard, speed=args.speed)
    model = None
    try:
        if args.resume:
            model = PPO.load(str(args.resume), env=env, device=args.device)
            model.set_random_seed(args.seed)
        else:
            model = PPO('MlpPolicy', env, n_steps=args.n_steps, batch_size=args.batch_size,
                        learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                        ent_coef=0.005, policy_kwargs={'net_arch': [128, 128]},
                        seed=args.seed, device=args.device, verbose=1)
        callbacks = [CheckpointCallback(save_freq=args.checkpoint_freq,
                     save_path=str(run_dir / 'checkpoints'), name_prefix='ppo'), ViewerCallback(env)]
        model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print('Durduruldu; son model kaydediliyor.', flush=True)
    finally:
        if model is not None:
            model.save(str(run_dir / 'final_model'))
            print(f'Model: {(run_dir / "final_model.zip").resolve()}', flush=True)
        env.close()


if __name__ == '__main__':
    main()

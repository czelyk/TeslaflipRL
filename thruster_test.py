"""Visible, measured nozzle tests. This free-space bench is separate from RL episodes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

from teslafliprl.config import EnvConfig
from teslafliprl.thruster_diagnostics import measure_all
from teslafliprl.vehicle import THRUSTERS, apply_thrusters, build_vehicle, update_thruster_visuals


def show_bench(manual=False, run_dir=Path('runs/thruster_test'), speed=1.):
    config = EnvConfig(randomize=False)
    client = p.connect(p.GUI)
    try:
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setTimeStep(1 / config.physics_hz, physicsClientId=client)
        p.resetDebugVisualizerCamera(5.6, 35, -25, [0, 0, 3], physicsClientId=client)
        car, wheels = build_vehicle(client, config, [0, 0, 3])
        p.setJointMotorControlArray(car, wheels, p.VELOCITY_CONTROL, forces=[0] * 4, physicsClientId=client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(manual), physicsClientId=client)
        selector = p.addUserDebugParameter('Kanal 0..12', 0, 12, 0, physicsClientId=client) if manual else None
        power = p.addUserDebugParameter('Guc yuzde', 0, 100, 100, physicsClientId=client) if manual else None
        p.addUserDebugText('ITICI TESTI | yercekimi kapali | RL bolumu degil', [-2, 0, 5.2], [1, 1, 1], 1.1, physicsClientId=client)
        if manual:
            p.addUserDebugText('BASILI TUT: SPACE=atesle | R=sifirla | slider=kanal/guc', [-2, 0, 4.8], [1, .8, .1], 1, physicsClientId=client)
        label, metrics = -1, -1
        state = None
        previous_index = -1
        started = time.perf_counter()
        tick = 0
        while p.isConnected(client):
            frame = time.perf_counter()
            keys = p.getKeyboardEvents(physicsClientId=client)
            if manual:
                index = min(len(THRUSTERS) - 1, max(0, int(round(p.readUserDebugParameter(selector, physicsClientId=client)))))
                firing = bool(keys.get(32, 0) & p.KEY_IS_DOWN)
                level = p.readUserDebugParameter(power, physicsClientId=client) / 100 if firing else 0.
                reset_requested = bool(keys.get(ord('r'), 0) & p.KEY_WAS_TRIGGERED)
            else:
                block = int(tick / (config.control_hz * 1.4))
                if block >= len(THRUSTERS):
                    break
                index = block
                level = 1. if tick % round(config.control_hz * 1.4) < config.control_hz * .6 else 0.
                reset_requested = False
            if index != previous_index or reset_requested:
                # Explicit diagnostic fixture reset only; production RL never uses this shortcut.
                p.resetBasePositionAndOrientation(car, [0, 0, 3], [0, 0, 0, 1], physicsClientId=client)
                p.resetBaseVelocity(car, [0, 0, 0], [0, 0, 0], physicsClientId=client)
                for wheel in wheels:
                    p.resetJointState(car, wheel, 0, 0, physicsClientId=client)
                previous_index = index
                state = None
                print(f'{index + 1}/13: {THRUSTERS[index].name}', flush=True)
            levels = np.zeros(len(THRUSTERS))
            levels[index] = level
            for _ in range(config.physics_hz // config.control_hz):
                apply_thrusters(client, car, THRUSTERS, levels, config.thruster_force)
                p.stepSimulation(physicsClientId=client)
            state = update_thruster_visuals(client, car, levels, state)
            position, _ = p.getBasePositionAndOrientation(car, physicsClientId=client)
            velocity, omega = p.getBaseVelocity(car, physicsClientId=client)
            p.resetDebugVisualizerCamera(5.6, 35, -25, position, physicsClientId=client)
            label = p.addUserDebugText(f'{index + 1}/13 {THRUSTERS[index].name}: {100 * level:.0f}% | {config.thruster_force * level:.0f} N',
                [-2, 0, 4.35], [1, .6, .1], 1.1, replaceItemUniqueId=label, physicsClientId=client)
            metrics = p.addUserDebugText(f'v={np.linalg.norm(velocity):.3f} m/s | omega={np.linalg.norm(omega):.3f} rad/s',
                [-2, 0, 4.0], [.1, 1, 1], 1, replaceItemUniqueId=metrics, physicsClientId=client)
            if not manual and index == 0 and tick == 25:
                from PIL import Image
                view = p.computeViewMatrixFromYawPitchRoll(position, 5.6, 35, -25, 0, 2)
                proj = p.computeProjectionMatrixFOV(55, 1.6, .1, 50)
                _, _, rgba, _, _ = p.getCameraImage(960, 600, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=client)
                Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(600, 960, 4)[:, :, :3]).save(run_dir / 'nozzle_firing.png')
            tick += 1
            delay = 1 / config.control_hz / speed - (time.perf_counter() - frame)
            if delay > 0:
                time.sleep(delay)
        return time.perf_counter() - started
    finally:
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


def main():
    parser = argparse.ArgumentParser(description='13 iticiyi tek tek olc ve goruntule.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--manual', action='store_true', help='Kanal/guc slider; SPACE basili=atesle, R=sifirla.')
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/thruster_test'))
    args = parser.parse_args()
    if args.speed <= 0 or (args.headless and args.manual):
        parser.error('speed pozitif olmali; --manual pencere gerektirir.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = measure_all()
    report = {'passed': all(row['passed'] for row in rows), 'thrusters_tested': len(rows),
              'fixture': 'complete rigid-body car, gravity=0, no contacts, wheel motors off', 'rows': rows}
    (args.run_dir / 'thruster_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['name']:14s}: {row['force_n']:.0f} N | dv={row['measured_delta_v_along_force_m_s']:.5f} m/s | omega={np.linalg.norm(row['angular_velocity_rad_s']):.5f} rad/s", flush=True)
    if not report['passed']:
        raise SystemExit(1)
    if not args.headless:
        show_bench(args.manual, args.run_dir, args.speed)
    print(f'13/13 itici testi gecti. Rapor: {args.run_dir / "thruster_report.json"}', flush=True)


if __name__ == '__main__':
    main()

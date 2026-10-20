"""Configuration and model metadata shared by train/watch/smoke."""
import json
from dataclasses import asdict
from pathlib import Path

from .config import EnvConfig


def load_config(model_path=None):
    if model_path:
        path = Path(model_path).resolve()
        for parent in [path.parent, path.parent.parent]:
            source = parent / 'env_config.json'
            if source.exists():
                return EnvConfig(**json.loads(source.read_text(encoding='utf-8')))
        raise FileNotFoundError('Model yanında veya üst klasörde env_config.json bulunamadı.')
    return EnvConfig()


def save_config(config, run_dir):
    target = Path(run_dir) / 'env_config.json'
    encoded = json.dumps(asdict(config), ensure_ascii=False, indent=2)
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != asdict(config):
        raise ValueError('Bu run-dir farklı fizik ayarları içeriyor; yeni bir klasör seçin.')
    target.write_text(encoded, encoding='utf-8')

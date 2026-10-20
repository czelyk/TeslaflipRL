"""Write portable version evidence without conda build-machine file URLs."""
import json
from importlib.metadata import distributions
from pathlib import Path


def main():
    versions = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
    Path('installed_versions.json').write_text(json.dumps(dict(sorted(versions.items())), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

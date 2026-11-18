import platform
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dir(tag):
    """Folder for one data size, e.g. results/n20000."""
    d = ROOT / "results" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


class Logger:
    """Prints and also writes to a log file, so every run leaves a record."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def versions(self):
        import numpy, pandas, sklearn, umap
        self(f"python {sys.version.split()[0]} on {platform.platform()}")
        self(f"numpy {numpy.__version__}, pandas {pandas.__version__}, "
             f"scikit-learn {sklearn.__version__}, umap-learn {umap.__version__}")

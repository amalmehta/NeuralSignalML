"""Where things live on disk. Everything is relative to the project folder."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = ROOT / "docs" / "figures"
CONFIGS_DIR = ROOT / "configs"

"""Project paths, independent of the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def resolve_path(path, base=PROJECT_ROOT):
    """Resolve relative paths against the project root or an explicit base."""
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path

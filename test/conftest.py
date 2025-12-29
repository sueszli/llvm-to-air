import sys
from pathlib import Path

# add project root to sys.path so we can import from src
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

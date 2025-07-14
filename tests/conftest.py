# conftest.py  (at project root)
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"

# Add src directory to Python path for proper imports
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
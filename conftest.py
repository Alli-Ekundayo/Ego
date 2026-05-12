# conftest.py
# Ensures the project root is on sys.path so all imports resolve
# when running: pytest tests/ -v
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

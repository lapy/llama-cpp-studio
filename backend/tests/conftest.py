"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Ensure backend is importable when running from repo root
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

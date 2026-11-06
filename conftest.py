"""Hace importables los paquetes ``core``, ``models`` y ``evaluation`` en pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

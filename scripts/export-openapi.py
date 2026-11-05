"""Write a deterministic OpenAPI document without starting application services."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STUDIO_DATA_DIR", tempfile.mkdtemp(prefix="studio-openapi-"))
os.environ.setdefault("STUDIO_ACCESS_MODE", "local")

from backend.main import app  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true", help="fail if the checked-in schema is stale")
arguments = parser.parse_args()

destination = ROOT / "frontend" / "src" / "api" / "openapi.json"
rendered = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
if arguments.check:
    current = destination.read_text(encoding="utf-8") if destination.exists() else ""
    if current != rendered:
        raise SystemExit("frontend/src/api/openapi.json is stale; run npm run openapi:export")
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
print(destination.relative_to(ROOT))

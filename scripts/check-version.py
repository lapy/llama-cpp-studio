"""Fail when package and backend version metadata diverge from VERSION."""

import json
import sys
from pathlib import Path


root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
expected = (root / "VERSION").read_text(encoding="utf-8").strip()
package = json.loads((root / "package.json").read_text(encoding="utf-8"))
if package.get("version") != expected:
    raise SystemExit(
        f"package.json version {package.get('version')!r} does not match VERSION {expected!r}"
    )

from backend.version import APP_VERSION

if APP_VERSION != expected:
    raise SystemExit(f"backend version {APP_VERSION!r} does not match VERSION {expected!r}")
print(expected)

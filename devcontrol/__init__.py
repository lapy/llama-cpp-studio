"""Development-only control plane for the isolated Studio dev stack.

This package is intentionally outside ``backend/``. The production image only
copies ``backend/``, ``frontend/public`` and the built frontend, so this code is
not importable from a production build.

The control plane is deliberately stdlib-only: it runs from a development
checkout over SSH and must not depend on the virtualenv used by the application.
"""

from __future__ import annotations

__version__ = "1.0.0"
PROTOCOL_VERSION = "1.0"

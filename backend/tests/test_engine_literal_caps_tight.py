"""R104 pin: the literal-ratchet caps are TIGHT, not merely ceilings.

``engine-literals-ratchet.py`` freezes per-area caps so growth requires a reviewed
diff. Slack in a cap is a silent allowance: the tree can grow up to the ceiling
without anyone editing the file. This pin asserts every cap EQUALS the measured
area total, so retiring literals forces the cap down in the same review.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/engine-literals-ratchet.py"


def _ratchet():
    spec = importlib.util.spec_from_file_location("literal_ratchet", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_area_caps_equal_the_measured_totals():
    ratchet = _ratchet()
    totals = ratchet.area_totals(ratchet.current_counts())
    caps = dict(ratchet.AREA_CAPS)
    assert set(totals) == set(caps), {
        "missing_cap": sorted(set(totals) - set(caps)),
        "stale_cap": sorted(set(caps) - set(totals)),
    }
    slack = {area: caps[area] - totals[area] for area in caps if caps[area] != totals[area]}
    assert slack == {}, f"caps with slack (tighten them): {slack}"


def test_ratchet_script_still_reports_success():
    ratchet = _ratchet()
    assert ratchet.main() == 0

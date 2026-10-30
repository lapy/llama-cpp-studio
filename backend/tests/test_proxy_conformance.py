import os

import pytest

from scripts.proxy_conformance import run


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("LLAMA_SWAP_TEST_BINARY"),
    reason="Pinned llama-swap binary is installed by the CI conformance step",
)
async def test_live_llama_swap_streaming_conformance(tmp_path):
    report_path = os.getenv("PROXY_CONFORMANCE_REPORT", str(tmp_path / "conformance.json"))
    from pathlib import Path

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report = await run(os.environ["LLAMA_SWAP_TEST_BINARY"], report_path)
    assert report["passed"], report
    assert len(report["checks"]) >= 9

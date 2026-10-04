"""Bounded workers for blocking Hugging Face metadata lookups.

These calls use the Hugging Face SDK, which performs synchronous network I/O.
They must not run on the API event loop. A small pool keeps a slow upstream
from occupying every worker the process uses for other blocking work.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

METADATA_WORKERS = 4
METADATA_TIMEOUT_SECONDS = 20.0

_executor = ThreadPoolExecutor(
    max_workers=METADATA_WORKERS,
    thread_name_prefix="hf-meta",
)
T = TypeVar("T")


async def run_metadata_lookup(func: Callable[..., T], /, *args: object) -> T:
    """Run ``func(*args)`` on the metadata pool and surface a deadline."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, func, *args),
        timeout=METADATA_TIMEOUT_SECONDS,
    )

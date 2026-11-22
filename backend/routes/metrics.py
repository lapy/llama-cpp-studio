"""Operator metrics endpoint (Prometheus text exposition)."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.observability.metrics import collect_metrics_text

router = APIRouter()


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    response_model=None,
    summary="Prometheus metrics for the management process",
)
async def get_metrics() -> PlainTextResponse:
    """Expose Studio's operational state as Prometheus text.

    Management-authorized like the rest of ``/api``. Values come from existing
    in-process state and cached observations; unknown values are omitted. The
    response is the Prometheus text format (version 0.0.4).
    """
    return PlainTextResponse(
        await collect_metrics_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )

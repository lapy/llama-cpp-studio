"""Run and compare bounded local model benchmarks."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.data_store import get_store
from backend.services.model_benchmark import (
    BenchmarkError,
    list_model_benchmarks,
    run_model_benchmark,
)


router = APIRouter()


class BenchmarkRunBody(BaseModel):
    model_id: str = Field(min_length=1, max_length=300)
    prompt: str = Field(
        default="Reply with a short description of the sky.", min_length=1, max_length=1000
    )
    max_tokens: int = Field(default=64, ge=1, le=512)


class BenchmarkResponse(BaseModel):
    id: str
    created_at: float
    model_id: str
    proxy_name: str
    config_revision: str
    config_fingerprint: str
    prompt: str
    max_tokens: int
    time_to_first_token_ms: float
    total_seconds: float
    completion_tokens: int | None
    tokens_per_second: float | None
    peak_observed_gpu_memory_bytes: int | None
    output_preview: str


def _error(exc: BenchmarkError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail},
    )


@router.post("/benchmarks/run", response_model=BenchmarkResponse)
async def run_benchmark(body: BenchmarkRunBody):
    try:
        return await run_model_benchmark(
            get_store(),
            body.model_id,
            prompt=body.prompt,
            max_tokens=body.max_tokens,
        )
    except BenchmarkError as exc:
        return _error(exc)


@router.get("/benchmarks/{model_id:path}", response_model=list[BenchmarkResponse])
def benchmark_history(model_id: str, limit: int = Query(default=20, ge=1, le=100)):
    return list_model_benchmarks(get_store(), model_id, limit=limit)

import asyncio

from fastapi import APIRouter, HTTPException

from backend.inference_auth import create_key, delete_key, list_keys, rotate_key, update_key_policy

router = APIRouter()


@router.get("")
async def get_inference_keys():
    from backend.proxy.studio.accounting import get_accounting

    def payload():
        accounting = get_accounting()
        return {"keys": [{**key, "usage": accounting.key_usage(key)} for key in list_keys()]}

    return await asyncio.to_thread(payload)


@router.post("", status_code=201)
async def post_inference_key(body: dict):
    try:
        record, token = create_key(
            body.get("name", ""),
            expires_at=body.get("expires_at"),
            models=body.get("models"),
            scopes=body.get("scopes"),
            rate_limit_per_minute=body.get("rate_limit_per_minute"),
            token_accounting=body.get("token_accounting", "auto"),
            **{
                name: body.get(name)
                for name in (
                    "input_tokens_per_minute",
                    "output_tokens_per_minute",
                    "daily_token_budget",
                    "monthly_token_budget",
                    "max_output_tokens",
                    "max_concurrent_requests",
                )
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**record, "key": token}


@router.post("/{key_id}/rotate")
async def rotate_inference_key(key_id: str):
    """Replace a key's secret; the old token stops working immediately."""
    result = rotate_key(key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference key not found")
    record, token = result
    return {**record, "key": token}


@router.delete("/{key_id}", status_code=204)
async def remove_inference_key(key_id: str):
    if not delete_key(key_id):
        raise HTTPException(status_code=404, detail="Inference key not found")
    return None


@router.patch("/{key_id}")
async def patch_inference_key(key_id: str, body: dict):
    try:
        from backend.proxy.studio.accounting import get_accounting

        # Publish completed usage before switching an unlimited key to a cap.
        await asyncio.to_thread(get_accounting().flush)
        result = await asyncio.to_thread(update_key_policy, key_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Inference key not found")
    return result

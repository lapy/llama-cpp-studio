from fastapi import APIRouter, HTTPException

from backend.inference_auth import create_key, delete_key, list_keys, rotate_key

router = APIRouter()


@router.get("")
async def get_inference_keys():
    return {"keys": list_keys()}


@router.post("", status_code=201)
async def post_inference_key(body: dict):
    try:
        record, token = create_key(
            body.get("name", ""),
            expires_at=body.get("expires_at"),
            models=body.get("models"),
            scopes=body.get("scopes"),
            rate_limit_per_minute=body.get("rate_limit_per_minute"),
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

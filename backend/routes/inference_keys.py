from fastapi import APIRouter, HTTPException

from backend.inference_auth import create_key, delete_key, list_keys

router = APIRouter()


@router.get("")
async def get_inference_keys():
    return {"keys": list_keys()}


@router.post("", status_code=201)
async def post_inference_key(body: dict):
    try:
        record, token = create_key(body.get("name", ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**record, "key": token}


@router.delete("/{key_id}", status_code=204)
async def remove_inference_key(key_id: str):
    if not delete_key(key_id):
        raise HTTPException(status_code=404, detail="Inference key not found")
    return None

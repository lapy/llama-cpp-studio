from typing import Dict, Any, Optional


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Safely convert value to int, returning default on failure."""
    try:
        return int(val)
    except (ValueError, TypeError, OverflowError):
        return default


def clamp_float(val: Any, lo: float, hi: float, default: float) -> float:
    try:
        fv = float(val)
    except Exception:
        return default
    return max(lo, min(hi, fv))


def clamp_int(val: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(val)
    except Exception:
        return default
    return max(lo, min(hi, iv))


def build_generation_params(
    architecture: str, context_length: int, preset_overrides: Dict[str, Any] | None
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    params.update(
        {
            "temperature": 0.8,
            "top_p": 0.9,
            "typical_p": 1.0,
            "min_p": 0.0,
            "tfs_z": 1.0,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "ctx_size": max(512, int(context_length or 0)),
            "stop": [],
        }
    )

    if preset_overrides:
        params.update(preset_overrides)

    params["temp"] = params.get("temperature", params.get("temp", 0.8))

    params["temperature"] = clamp_float(params.get("temperature", 0.8), 0.0, 2.0, 0.8)
    params["top_p"] = clamp_float(params.get("top_p", 0.9), 0.0, 1.0, 0.9)
    params["min_p"] = clamp_float(params.get("min_p", 0.0), 0.0, 1.0, 0.0)
    params["typical_p"] = clamp_float(params.get("typical_p", 1.0), 0.0, 1.0, 1.0)
    params["tfs_z"] = clamp_float(params.get("tfs_z", 1.0), 0.0, 1.0, 1.0)
    params["top_k"] = max(0, int(params.get("top_k", 40) or 0))
    params["repeat_penalty"] = max(0.0, float(params.get("repeat_penalty", 1.1) or 1.1))
    params["presence_penalty"] = float(params.get("presence_penalty", 0.0) or 0.0)
    params["frequency_penalty"] = float(params.get("frequency_penalty", 0.0) or 0.0)
    params["mirostat"] = max(0, min(2, int(params.get("mirostat", 0) or 0)))
    params["mirostat_tau"] = clamp_float(
        params.get("mirostat_tau", 5.0), 0.1, 20.0, 5.0
    )
    params["mirostat_eta"] = clamp_float(
        params.get("mirostat_eta", 0.1), 0.01, 2.0, 0.1
    )
    params["ctx_size"] = max(
        512, int(params.get("ctx_size", context_length) or context_length)
    )
    if not isinstance(params.get("stop", []), list):
        params["stop"] = []

    return params

"""llama.cpp family: llama_cpp and ik_llama."""

__all__ = ["BuildConfig", "LlamaCppManager", "LlamaManager"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from backend.engines.llama_cpp.manager import BuildConfig, LlamaCppManager, LlamaManager

    return {"BuildConfig": BuildConfig, "LlamaCppManager": LlamaCppManager, "LlamaManager": LlamaManager}[name]

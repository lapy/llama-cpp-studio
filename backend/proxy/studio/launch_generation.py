"""Select and publish the launch generation used by a new engine process.

Process creation uses the model's published generation. Compiling the current
desired configuration at launch time would let an engine install (which
activates the new version) silently replace a running model: the next request
compiled a different revision, the registry started a second engine next to the
loaded one, and for GPU engines that second start failed (typically OOM) -
surfacing as 502 for every request while the original process stayed healthy.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from backend.proxy.launch_spec import LaunchSpec, compile_model_runtime, manifest_document


async def launch_for_process(
    manifests: Any,
    model_id: str,
    load_model: Callable[[str], Optional[Dict[str, Any]]],
):
    """Return ``(LaunchSpec, revision, artifact_dir)`` for a process creation.

    A published pointer selects the generation. The first genuine start has no
    pointer yet, so it publishes exactly what it is about to run; later desired
    edits (including an engine-version activation) wait for an explicit apply.
    """
    published = manifests.read_pointer(model_id)
    if published is not None:
        document = manifests.read_manifest(model_id, published.revision)
        launch = LaunchSpec.from_document(document)
        if launch.model_id != model_id:
            raise ValueError(f"published manifest is for {launch.model_id!r}, not {model_id!r}")
        generation = manifests.generation_dir(model_id, published.revision)
        return launch, published.revision, os.path.join(generation, "artifacts")

    model = await asyncio.to_thread(load_model, model_id)
    if model is None:
        raise ValueError(f"unknown model: {model_id}")
    compiled = await asyncio.to_thread(compile_model_runtime, model)
    document = manifest_document(
        compiled.launch,
        compiled.revision,
        profile_digest=compiled.profile_digest,
    )
    manifests.stage(compiled.proxy.model_id, document)
    manifests.publish(compiled.proxy.model_id, compiled.revision)
    generation = manifests.generation_dir(compiled.proxy.model_id, compiled.revision)
    return compiled.launch, compiled.revision, os.path.join(generation, "artifacts")

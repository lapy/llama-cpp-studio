# Reference definitions — schema_version 2 spike

These two files are the falsification test required by
[the refactoring plan](../../../docs/config-driven-engine-refactoring-plan.md):
before Phase 3 builds any executor, prove the schema can express **one best
case** (`lmdeploy`: the simplest python_venv engine) and the **worst case**
(`sglang_v100`: SM70 specialization, Studio-managed CUDA 12.8, source
patches, memory-aware build concurrency, and two runtime profiles).

They are **not loaded by the runtime** (the definition loader reads only the
parent directory; the shadow-mode guard test proves nothing imports them)
and they stay **descriptor-equivalent** to `ENGINE_REGISTRY` — enforced by
`backend/tests/test_engine_definitions_v2.py`, along with golden argv/env
checks against today's `backend/proxy/launch_spec.py` behavior.

Values cite their current source of truth in comments. If legacy behavior
changes, update the reference file and its golden test together.

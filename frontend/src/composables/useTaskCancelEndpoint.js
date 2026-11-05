/** Resolve the manager-specific cancel API for a progress task. */

import {
  DECLARED_CANCEL_ENDPOINTS_BY_ENGINE,
  NATIVE_CANCEL_ENDPOINTS_BY_ENGINE,
  serverCancelEndpoint,
  VENV_CANCEL_ENDPOINTS_BY_MANAGER,
} from '../engines/registry.js'

const MANAGER_CANCEL_ENDPOINTS = {
  cuda: '/api/llama-versions/cuda/cancel',
  // Native engines resolve through the engine registry (generic
  // /api/engines/{engine_id}/cancel router; byte-equivalence pinned by
  // backend/tests/test_native_engine_lifecycle.py). Covers the
  // unsloth_llama manager id and the llama_cpp/ik_llama engine ids that
  // cmake build tasks carry in metadata.engine.
  ...NATIVE_CANCEL_ENDPOINTS_BY_ENGINE,
  // R79: engines with their own cancel API declare `cancelPath` on the
  // descriptor (audio.cpp) instead of a literal map entry here.
  ...DECLARED_CANCEL_ENDPOINTS_BY_ENGINE,
  // Python-venv managers resolve through the engine registry (generic
  // /api/engines/{engine_id}/cancel router; byte-equivalence pinned by
  // backend/tests/test_generic_engine_router.py).
  ...VENV_CANCEL_ENDPOINTS_BY_MANAGER,
}

const TYPE_CANCEL_ENDPOINTS = {
  // Legacy fallback for build tasks without engine metadata (pre-registry
  // task shapes); tasks that carry engine ids resolve above.
  build: '/api/llama-versions/build-cancel',
  download: '/api/models/downloads/cancel',
}

export function cancelEndpointForTask(task) {
  if (!task) return null
  if (task.task_id && (task.type === 'audio_model_install' || task.type === 'audio_model_import')) {
    return `/api/model-catalog/tasks/${encodeURIComponent(task.task_id)}/cancel`
  }
  const manager = task.metadata?.manager
  if (manager && MANAGER_CANCEL_ENDPOINTS[manager]) {
    return MANAGER_CANCEL_ENDPOINTS[manager]
  }
  // YAML campaign S5: YAML-onboarded engines cancel through the generic
  // router, discovered from the live descriptor payload — no table entry.
  if (manager && serverCancelEndpoint(manager)) {
    return serverCancelEndpoint(manager)
  }
  // Native cmake builds share type=build; route audio.cpp to its cancel API.
  const engine = task.metadata?.engine
  if (engine && MANAGER_CANCEL_ENDPOINTS[engine]) {
    return MANAGER_CANCEL_ENDPOINTS[engine]
  }
  if (engine && serverCancelEndpoint(engine)) {
    return serverCancelEndpoint(engine)
  }
  if (task.type && TYPE_CANCEL_ENDPOINTS[task.type]) {
    return TYPE_CANCEL_ENDPOINTS[task.type]
  }
  return null
}

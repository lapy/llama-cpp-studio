/**
 * Engine registry — single source of truth for engine identity on the frontend.
 *
 * Part of the config-driven engine refactor
 * (docs/config-driven-engine-refactoring-plan.md). Mirrors the backend
 * ``LIFECYCLE`` registry (backend/routes/engine_versions.py) and
 * docs/engine-surface-contract-matrix.md.
 *
 * v3 (identity convergence): the nine engine descriptors also absorb what
 * used to be six parallel engine-id-keyed records scattered across views:
 * display labels + icons (fallbackEngineOptions, ENGINE_LABELS), build-hint
 * copy (EngineBuildSettingsHint), the cmake-build family (VersionTable's
 * CMAKE_ENGINES), the sglang-runtime-UI family (ModelConfig's
 * SGLANG_ENGINES — deliberately WITHOUT vllm), the installer panel family
 * (WITH vllm), and the client fallback artifact-format map
 * (inferredEnginesByFormat). Consumers derive from here; add an engine by
 * adding ONE descriptor.
 *
 * API routing: descriptors with ``genericApi: true`` (the deliberate
 * migration allow-list — backend equivalence pinned by
 * backend/tests/test_generic_engine_router.py) resolve through the generic
 * ``/api/engines/{engine_id}/{verb}`` router; others keep the legacy
 * ``/api/{apiSegment}/{verb}`` prefix. ``legacyVenvApiPath`` keeps the
 * legacy segments as the pinned rollback path. The native llama-family
 * engines are not venv descriptors: their generic verbs route through
 * ``NATIVE_GENERIC_VERBS``/``nativeGenericApiPath`` (a client-side mirror
 * of the backend verb maps — unlisted verbs 405 server-side, so no URL is
 * ever built for them; equivalence pinned by
 * backend/tests/test_native_engine_lifecycle.py).
 *
 * Descriptors carry the REAL per-engine divergences the matrix records
 * (§6) as data so callers cannot accidentally "unify" them:
 * - ``nullSafeStatus``: the SGLang-family producers assign ``data || {}``
 *   to the keyed status map (and return it); lmdeploy/1Cat assign the raw
 *   payload. R81 replaced the per-engine ``statusRef`` names with one
 *   engine-id-keyed store map, so this flag is the only status divergence.
 * - ``managerKey``: the ``task.metadata.manager`` id used by the task-cancel
 *   map (differs from the engine id for 1Cat-vLLM).
 * - ``repoSource``: the ``repository_source`` display name the version rows
 *   carry (the SECOND identity system — owned here, matched nowhere else).
 */

export const ENGINE_DESCRIPTORS = Object.freeze({
  llama_cpp: Object.freeze({
    label: 'llama.cpp',
    card: Object.freeze({
      mark: Object.freeze({ text: 'L', className: 'engine-mark--llama' }),
      subtitle: 'GGUF on CPU or GPU.',
      update: Object.freeze({ key: 'llamaCpp', formatter: 'plain' }),
    }),
    repoSource: 'llama.cpp',
    versionsKey: 'llamaVersions',
    // R95: the build-options catalog endpoint is declared here; the panel
    // dispatches through `buildOptionsPath()` instead of engine-id branches.
    buildOptionsPath: '/api/llama-versions/build-options',
    // Legacy rows predate repository_source tagging and belong to llama.cpp.
    untaggedSourceOwner: true,
    icon: 'pi-microchip',
    cmakeCapable: true,
    fallbackFormats: Object.freeze(['gguf']),
    buildHint:
      'Choose CUDA or CPU, server binary artifacts, and other CMake options before you install or build. Reopen Build settings from the dialog header anytime.',
    buildUi: Object.freeze({
      refPlaceholder: 'master',
      refHint:
        'Stable tags are vX.Y.Z. Nightly tags are still bNNNN (the master commit count, not a commit SHA). A branch or commit also works.',
    }),
  }),
  ik_llama: Object.freeze({
    label: 'ik_llama.cpp',
    card: Object.freeze({
      mark: Object.freeze({ text: 'IK', className: 'engine-mark--ik' }),
      subtitle: 'GGUF, including IQK quants.',
      update: Object.freeze({ key: 'ikLlama', formatter: 'plain' }),
    }),
    repoSource: 'ik_llama.cpp',
    versionsKey: 'ikLlamaVersions',
    buildOptionsPath: '/api/llama-versions/build-options',
    icon: 'pi-microchip',
    cmakeCapable: true,
    fallbackFormats: Object.freeze(['gguf']),
    buildHint:
      'Choose CUDA or CPU, IQK options, and other CMake flags before you build. ik_llama.cpp tracks the tip of main (no release tags). Reopen Build settings from the dialog header anytime.',
    buildUi: Object.freeze({
      refPlaceholder: 'main or commit SHA',
      refHint:
        'Use a branch or commit. This engine does not ship releases or tags here; check for updates and "build latest" track the tip of main.',
      backendNote:
        'This engine uses IQK kernels and GGML_CUDA_USE_GRAPHS naming. Examples must stay on (server lives there).',
      artifactsNote: 'For this engine, Examples is required (server binary lives in examples).',
      disableBuildExamples: true,
    }),
  }),
  unsloth_llama: Object.freeze({
    label: 'Unsloth llama.cpp',
    card: Object.freeze({
      mark: Object.freeze({ text: 'US', className: 'engine-mark--unsloth' }),
      subtitle: 'GGUF with Unsloth llama-server.',
      update: Object.freeze({ key: 'unslothLlama', formatter: 'plain' }),
    }),
    repoSource: 'Unsloth llama.cpp',
    versionsKey: 'unslothLlamaVersions',
    icon: 'pi-microchip',
    fallbackFormats: Object.freeze(['gguf']),
    // No build-hint copy: unsloth ships release installs only.
  }),
  lmdeploy: Object.freeze({
    label: 'LMDeploy',
    card: Object.freeze({
      mark: Object.freeze({ icon: 'pi pi-server' }),
      subtitle: 'Safetensors via a Python environment.',
      update: Object.freeze({ key: 'lmdeploy', formatter: 'v' }),
    }),
    repoSource: 'LMDeploy',
    icon: 'pi-server',
    fallbackFormats: Object.freeze(['safetensors']),
    venv: true,
    apiSegment: 'lmdeploy',
    versionsKey: 'lmdeployVersions',
    dialogIcon: 'pi-server',
    genericApi: true,
    managerKey: 'lmdeploy',
    buildHint:
      'Set default PyPI version and source repo/branch before installing. Reopen Build settings from the dialog header anytime.',
  }),
  '1cat_vllm': Object.freeze({
    label: '1Cat-vLLM',
    card: Object.freeze({
      mark: Object.freeze({ icon: 'pi pi-bolt' }),
      subtitle: 'Safetensors on SM70 GPUs.',
      update: Object.freeze({ key: 'onecatVllm', formatter: 'v' }),
    }),
    repoSource: '1Cat-vLLM',
    icon: 'pi-server',
    fallbackFormats: Object.freeze(['safetensors']),
    venv: true,
    apiSegment: '1cat-vllm',
    versionsKey: 'onecatVllmVersions',
    // Dialog-header icon differs from the picker icon (pi-server) — real
    // per-surface divergence, kept as data.
    dialogIcon: 'pi-bolt',
    genericApi: true,
    managerKey: 'onecat_vllm',
    buildHint:
      'Set default release version and source repo/branch before installing or building. Reopen Build settings from the dialog header anytime.',
  }),
  sglang: Object.freeze({
    label: 'SGLang',
    panel: Object.freeze({
      variant: 'sglang',
      defaultRepo: 'https://github.com/sgl-project/sglang.git',
      projectUrl: 'https://pypi.org/project/sglang/',
    }),
    card: Object.freeze({
      mark: Object.freeze({ icon: 'pi pi-sparkles' }),
      subtitle: 'Safetensors, OpenAI-compatible API.',
    }),
    repoSource: 'SGLang',
    versionsKey: 'sglangVersions',
    icon: 'pi-sparkles',
    fallbackFormats: Object.freeze(['safetensors']),
    venv: true,
    apiSegment: 'sglang',
    nullSafeStatus: true,
    dialogIcon: 'pi-sparkles',
    genericApi: true,
    managerKey: 'sglang',
    sglangRuntimeUi: true,
    panelFamily: 'sglang',
    runtimeProfiles: Object.freeze({ llm: 'default', diffusion: 'diffusion' }),
    runtimeKey: 'sglang_runtime',
    buildHint:
      'Set the default PyPI version and source repo/branch before installing. Upstream SGLang currently requires a CUDA 13-capable environment.',
  }),
  sglang_v100: Object.freeze({
    label: 'SGLang V100',
    panel: Object.freeze({
      variant: 'v100',
      defaultRepo: 'https://github.com/haohervchb/sglang-V100.git',
      projectUrl: 'https://github.com/haohervchb/sglang-V100',
    }),
    card: Object.freeze({
      mark: Object.freeze({ icon: 'pi pi-bolt' }),
      badge: Object.freeze({ value: 'SM70', severity: 'warn' }),
      subtitle: 'V100 / SM70 source build.',
    }),
    repoSource: 'SGLang-V100',
    versionsKey: 'sglangV100Versions',
    icon: 'pi-sparkles',
    fallbackFormats: Object.freeze(['safetensors']),
    venv: true,
    apiSegment: 'sglang-v100',
    nullSafeStatus: true,
    dialogIcon: 'pi-bolt',
    genericApi: true,
    managerKey: 'sglang_v100',
    sglangRuntimeUi: true,
    panelFamily: 'sglang',
    runtimeProfiles: Object.freeze({ llm: 'default', diffusion: 'diffusion' }),
    runtimeKey: 'sglang_runtime',
    buildHint:
      'Set the V100 fork repo and branch. Install Studio-managed CUDA 12.8 first; the fork then builds only its SM70-specific dependencies.',
  }),
  vllm: Object.freeze({
    label: 'vLLM',
    panel: Object.freeze({
      variant: 'vllm',
      defaultRepo: 'https://github.com/vllm-project/vllm.git',
      projectUrl: 'https://pypi.org/project/vllm/',
    }),
    card: Object.freeze({
      mark: Object.freeze({ icon: 'pi pi-server' }),
      subtitle: 'Safetensors, OpenAI-compatible API.',
    }),
    repoSource: 'vLLM',
    versionsKey: 'vllmVersions',
    icon: 'pi-server',
    fallbackFormats: Object.freeze(['safetensors']),
    venv: true,
    apiSegment: 'vllm',
    nullSafeStatus: true,
    // Header icon is pi-sparkles, unlike the picker's pi-server — and the
    // V100 sibling reuses pi-bolt. Kept as data, not "unified".
    dialogIcon: 'pi-sparkles',
    genericApi: true,
    managerKey: 'vllm',
    panelFamily: 'sglang', // one installer panel family — WITHOUT the sglang-runtime UI
    buildHint:
      'Set the default PyPI version and source repo/branch. Source builds use the active Studio-managed CUDA toolkit.',
  }),
  audio_cpp: Object.freeze({
    label: 'audio.cpp',
    card: Object.freeze({
      mark: Object.freeze({ text: 'A', className: 'engine-mark--audio' }),
      badge: Object.freeze({ kind: 'audio-maturity' }),
      subtitle: 'Speech, music, and other audio tasks.',
      update: Object.freeze({ key: 'audioCpp', formatter: 'version' }),
      featureFlag: 'audio_cpp',
    }),
    repoSource: 'audio.cpp',
    versionsKey: 'audioCppVersions',
    buildOptionsPath: '/api/audio-cpp/build-options',
    // R101: ModelConfig renders the audio panel for this family (declared here
    // instead of comparing engine ids at every site).
    configPanel: 'audio',
    icon: 'pi-volume-up',
    cmakeCapable: true,
    // R79: audio.cpp's cancel API is its own endpoint (declared here instead of
    // a literal map in useTaskCancelEndpoint); backend route is pinned by
    // backend/tests/test_audio_protocol_contract.py (declared extension).
    cancelPath: '/api/audio-cpp/cancel',
    // No fallbackFormats: curated audio packages are gated by package_kind,
    // not by artifact format (see inferredEnginesForModel).
    configCopy: Object.freeze({
      stableIdLabel: 'Model ID while running',
      stableIdAria: 'Stable model ID',
      aliasLabel: 'Friendly API name',
      aliasTooltip:
        'Optional name apps send as model. Running state uses the stable ID.',
      unknownKeysPrefix: 'Unrecognized audio.cpp keys are preserved on save:',
    }),
  }),
})

const byField = (field, value) =>
  Object.freeze(
    Object.keys(ENGINE_DESCRIPTORS).filter(
      (id) => ENGINE_DESCRIPTORS[id][field] === (value === undefined ? true : value),
    ),
  )

/** Engine ids served by the shared venv lifecycle (five python engines). */
export const VENV_ENGINE_IDS = byField('venv')

export const VENV_DESCRIPTORS = Object.freeze(
  Object.fromEntries(VENV_ENGINE_IDS.map((id) => [id, ENGINE_DESCRIPTORS[id]])),
)

/** Display metadata for the engine picker when server descriptors are absent. */
export const ENGINE_LABELS = Object.freeze(
  Object.fromEntries(Object.entries(ENGINE_DESCRIPTORS).map(([id, d]) => [id, d.label])),
)

export const FALLBACK_ENGINE_OPTIONS = Object.freeze(
  Object.entries(ENGINE_DESCRIPTORS).map(([value, d]) =>
    Object.freeze({ value, label: d.label, icon: d.icon }),
  ),
)

/** Client fallback for format→engine inference (server descriptors win). */
const withFormat = (fmt) =>
  Object.freeze(
    Object.keys(ENGINE_DESCRIPTORS).filter((id) =>
      ENGINE_DESCRIPTORS[id].fallbackFormats?.includes(fmt),
    ),
  )

export const ENGINE_IDS_BY_FALLBACK_FORMAT = Object.freeze({
  gguf: withFormat('gguf'),
  safetensors: withFormat('safetensors'),
})

/** Engines editable via the cmake build-config dialog (VersionTable). */
export const CMAKE_ENGINE_IDS = byField('cmakeCapable')

/** SGLang-runtime-UI family (ModelConfig) — deliberately excludes vllm. */
export const SGLANG_RUNTIME_UI_IDS = byField('sglangRuntimeUi')

/** Labels for the SGLang runtime selector; UI copy lives with the descriptors. */
export const SGLANG_RUNTIME_OPTIONS = Object.freeze([
  Object.freeze({ label: 'Language model', value: 'llm' }),
  Object.freeze({ label: 'Image / video (diffusion)', value: 'diffusion' }),
])

/** Audio config-panel family; no view compares an engine id by hand. */
export const AUDIO_ENGINE_IDS = byField('configPanel', 'audio')

/** Shared installer-panel family (SglangEnginePanel/store trio) — includes vllm. */
export const SG_PANEL_ENGINE_IDS = byField('panelFamily', 'sglang')

function engineDescriptor(engineId) {
  return Object.hasOwn(ENGINE_DESCRIPTORS, engineId) ? ENGINE_DESCRIPTORS[engineId] : null
}

export function engineBuildHint(engineId) {
  return engineDescriptor(engineId)?.buildHint || ''
}

export function venvDescriptor(engineId) {
  const descriptor = Object.hasOwn(VENV_DESCRIPTORS, engineId) ? VENV_DESCRIPTORS[engineId] : null
  if (!descriptor) {
    throw new Error(`Unknown venv engine: ${engineId}`)
  }
  return descriptor
}

export function legacyVenvApiPath(engineId, verb) {
  return `/api/${venvDescriptor(engineId).apiSegment}/${verb}`
}

export function venvApiPath(engineId, verb) {
  const descriptor = venvDescriptor(engineId)
  return descriptor.genericApi
    ? `/api/engines/${engineId}/${verb}`
    : legacyVenvApiPath(engineId, verb)
}

/**
 * R95: the build-options catalog endpoint an engine declares, or null when it
 * has no catalog (callers fall back to their legacy dispatch).
 */
export function buildOptionsPath(engineId) {
  const descriptor = ENGINE_DESCRIPTORS[engineId]
  return descriptor?.buildOptionsPath || null
}

/**
 * R101: the audio CONFIG PANEL family (an audio model gets the audio panel).
 * Declared per descriptor (`configPanel: 'audio'`) instead of an engine-id
 * comparison at every ModelConfig site.
 */
export function isAudioEngineId(engineId) {
  return ENGINE_DESCRIPTORS[engineId]?.configPanel === 'audio'
}

/** Reverse map for task-completion code that only knows manager ids.
 * Internal: consumers use engineIdForManager/engineLabel (hasOwn-hardened). */
const ENGINE_ID_BY_MANAGER = Object.freeze(
  Object.fromEntries(
    Object.entries(ENGINE_DESCRIPTORS)
      .filter(([, d]) => d.managerKey)
      .map(([id, d]) => [d.managerKey, id]),
  ),
)

export function engineIdForManager(manager) {
  return Object.hasOwn(ENGINE_ID_BY_MANAGER, manager) ? ENGINE_ID_BY_MANAGER[manager] : null
}

/* ---------------------------------------------------------------------
 * YAML campaign S5 — server-backed overlay (module-level derived view).
 *
 * fetchEngineDescriptors() feeds applyServerDescriptors() the raw
 * GET /api/engines payload; label/cancel lookups then see YAML-onboarded
 * engines with zero local table entries. Pure lookups stay pure — the
 * nine generated engines resolve IDENTICALLY through either source
 * (backend registry-equivalence pins server label == ENGINE_LABELS), so
 * cached values never change rendered bytes. Offline (no fetch yet) the
 * maps are empty and every lookup falls back to the table below.
 * ------------------------------------------------------------------- */
const serverLabelById = new Map()
const serverCancelEngines = new Set()

export function applyServerDescriptors(descriptors) {
  serverLabelById.clear()
  serverCancelEngines.clear()
  for (const descriptor of Array.isArray(descriptors) ? descriptors : []) {
    if (!descriptor || !descriptor.id) continue
    if (descriptor.label) serverLabelById.set(descriptor.id, descriptor.label)
    if (
      descriptor.generic_lifecycle === true &&
      (descriptor.generic_verbs || []).includes('cancel')
    )
      serverCancelEngines.add(descriptor.id)
  }
}

export function serverEngineLabel(engineId) {
  return serverLabelById.get(engineId) || null
}

export function serverCancelEndpoint(engineId) {
  return serverCancelEngines.has(engineId) ? `/api/engines/${engineId}/cancel` : null
}

export function engineLabel(managerOrEngineId) {
  const engineId = engineIdForManager(managerOrEngineId) || managerOrEngineId
  const serverLabel = serverLabelById.get(engineId)
  if (serverLabel) return serverLabel
  const label = Object.hasOwn(ENGINE_LABELS, engineId) ? ENGINE_LABELS[engineId] : null
  return label || engineId || 'another engine'
}

/** Task-cancel URLs keyed by ``task.metadata.manager`` (useTaskCancelEndpoint). */
export const VENV_CANCEL_ENDPOINTS_BY_MANAGER = Object.freeze(
  Object.fromEntries(
    VENV_ENGINE_IDS.filter((id) => ENGINE_DESCRIPTORS[id].genericApi).map((engineId) => [
      ENGINE_DESCRIPTORS[engineId].managerKey,
      `/api/engines/${engineId}/cancel`,
    ]),
  ),
)

/**
 * Verbs each native llama-family engine genuinely serves on the generic
 * ``/api/engines/{id}`` router (mirrors the backend ``verbs`` override maps;
 * unlisted verbs 405 server-side — see matrix §8 "Backend native
 * adapters"). The store routes every verb these engines implement through
 * nativeGenericApiPath.
 */
export const NATIVE_GENERIC_VERBS = Object.freeze({
  llama_cpp: Object.freeze([
    'check-updates',
    'build-settings',
    'install',
    'install-source',
    'cancel',
  ]),
  ik_llama: Object.freeze([
    'check-updates',
    'build-settings',
    'install',
    'install-source',
    'cancel',
  ]),
  unsloth_llama: Object.freeze(['check-updates', 'install', 'cancel']),
})

export function nativeGenericApiPath(engineId, verb) {
  if (
    !Object.hasOwn(NATIVE_GENERIC_VERBS, engineId) ||
    !NATIVE_GENERIC_VERBS[engineId].includes(verb)
  ) {
    throw new Error(`Generic API not available for engine: ${engineId}/${verb}`)
  }
  return `/api/engines/${engineId}/${verb}`
}

/**
 * Task-cancel URLs for native engines, keyed by both metadata surfaces a
 * task can carry: ``metadata.engine`` (llama.cpp / ik_llama cmake builds)
 * and ``metadata.manager`` (unsloth manager id == engine id). llama and
 * ik share one BuildTaskManager, so their per-engine URLs are routing —
 * both overrides delegate to the same cancel implementation.
 */
export const NATIVE_CANCEL_ENDPOINTS_BY_ENGINE = Object.freeze(
  Object.fromEntries(
    Object.keys(NATIVE_GENERIC_VERBS)
      .filter((engineId) => NATIVE_GENERIC_VERBS[engineId].includes('cancel'))
      .map((engineId) => [engineId, `/api/engines/${engineId}/cancel`]),
  ),
)

/**
 * Task-cancel URLs for engines whose cancel API is NOT the generic router
 * (R79): declared on the descriptor as ``cancelPath``. No per-engine literal
 * map in the composable — adding such an engine is one descriptor field.
 */
export const DECLARED_CANCEL_ENDPOINTS_BY_ENGINE = Object.freeze(
  Object.fromEntries(
    Object.entries(ENGINE_DESCRIPTORS)
      .filter(([, descriptor]) => descriptor.cancelPath)
      .map(([engineId, descriptor]) => [engineId, descriptor.cancelPath]),
  ),
)

/* ---------------------------------------------------------------------
 * YAML campaign S5 — server-backed descriptors (additive overlay).
 *
 * Engines onboarded purely through YAML are registered server-side
 * (backend startup); GET /api/engines marks them with
 * ``generic_lifecycle: true`` + ``generic_verbs``. These PURE helpers
 * turn such a descriptor into the shape the store's generic venv helpers
 * consume — NO local table entry, so a new YAML engine needs zero
 * frontend edits. The nine generated engines NEVER route through here:
 * their hardcoded descriptors above keep byte-identical behavior
 * (registry unit pins + engines.v3 parity guard). The local table is
 * demoted to offline fallback only for engines the server does not
 * describe; it is consulted nowhere for server-backed ids.
 * ------------------------------------------------------------------- */

export function serverBackedEngines(descriptors) {
  if (!Array.isArray(descriptors)) return []
  return descriptors.filter((d) => d && d.generic_lifecycle === true)
}

/**
 * Default engine for an artifact format, from descriptor ``fallbackFormats``.
 * Keeps catalog/library defaults out of engine-id lists.
 */
export function defaultEngineForFormat(format, descriptors = ENGINE_DESCRIPTORS) {
  const wanted = String(format || '').toLowerCase()
  if (!wanted) return ''
  for (const [engineId, descriptor] of Object.entries(descriptors || {})) {
    if ((descriptor?.fallbackFormats || []).includes(wanted)) return engineId
  }
  return ''
}

/** Descriptor label lookup with the engine id as a stable fallback. */

/**
 * Synthesize the store-side descriptor for a server-backed engine, or null
 * when the id is not server-backed. ``versionsKey`` is a DYNAMIC marker — the
 * store keeps server-backed rows and statuses in id-keyed maps instead of named
 * refs. ``managerKey`` equals the engine id (the generic installer's
 * MANAGER_NAME convention).
 */
export function serverBackedDescriptor(engineId, descriptors) {
  const found = Array.isArray(descriptors)
    ? descriptors.find((d) => d && d.id === engineId && d.generic_lifecycle === true)
    : null
  if (!found) return null
  return Object.freeze({
    label: found.label || engineId,
    repoSource: found.repository_source || '',
    icon: found.install_kind === 'python_venv' ? 'pi-server' : 'pi-box',
    venv: found.install_kind === 'python_venv',
    genericApi: true,
    nullSafeStatus: true,
    versionsKey: engineId,
    managerKey: engineId,
    genericVerbs: Object.freeze([...(found.generic_verbs || [])]),
    serverBacked: true,
  })
}

/** URL for a server-backed engine: the generic router ONLY. */
export function serverBackedApiPath(engineId, verb) {
  return `/api/engines/${encodeURIComponent(engineId)}/${verb}`
}

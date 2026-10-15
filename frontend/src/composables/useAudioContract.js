import { ref } from 'vue'
import { sessionFetch } from '@/api/sessionFetch.js'

/**
 * Single client-side view of the composed audio.cpp integration contract
 * (GET /api/audio-cpp/contract). Task→operation routing, public endpoint and
 * defaults-key tables come from here — never re-derived from task/family
 * strings in the frontend.
 */

export const AUDIO_CONTRACT_SCHEMA = 'audio-cpp.studio-contract/v0'

/**
 * The only offline facts kept on the client: Studio's generic task route and
 * the Studio-owned defaults buckets. With no operations table loaded, every
 * task falls through to the generic route — exactly like the backend registry
 * does for unknown tasks. All real routing arrives from the live payload.
 */
export const OFFLINE_AUDIO_CONTRACT = {
  schema: AUDIO_CONTRACT_SCHEMA,
  generic_studio_path: '/audioapi/v1/tasks/run',
  generic_native_path: '/v1/tasks/run',
  defaults_keys: { tasks_run: 'task_defaults' },
  task_routing: {},
  operations: [],
  family_routing: {},
  drift_warnings: [],
  engine_version: '',
}

export const audioContract = ref({ ...OFFLINE_AUDIO_CONTRACT })
export const audioContractLoaded = ref(false)
export const audioContractError = ref('')

let inflight = null

/** Fetch (or reuse) the composed contract for the active engine version. */
export async function fetchAudioContract({ force = false } = {}) {
  if (inflight && !force) return inflight
  const request = (async () => {
    try {
      const response = await sessionFetch('/api/audio-cpp/contract', {
        headers: { Authorization: 'Bearer local' },
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.detail || `HTTP ${response.status}`)
      }
      const data = await response.json()
      audioContract.value = { ...OFFLINE_AUDIO_CONTRACT, ...data }
      audioContractError.value = ''
      audioContractLoaded.value = true
      return audioContract.value
    } catch (error) {
      // 409 = no active modern engine yet; keep the offline table so forms can
      // render defaults, and surface the update prompt via the error text.
      audioContractError.value = error?.message || String(error)
      return audioContract.value
    } finally {
      if (inflight === request) inflight = null
    }
  })()
  inflight = request
  return request
}

function operationsBy() {
  const byId = new Map()
  for (const op of audioContract.value.operations || []) byId.set(op.id, op)
  return byId
}

/** The registry operation serving a task (deterministic table from backend). */
export function audioOperationForTask(task) {
  const contract = audioContract.value
  const key = String(task || '').trim().toLowerCase()
  const byId = operationsBy()
  const opId = key ? (contract.task_routing || {})[key] : null
  return (opId && byId.get(opId)) || byId.get('tasks_run') || null
}

/** Public (llama-swap) endpoint a task's requests go to. */
export function audioTaskEndpoint(task) {
  const op = audioOperationForTask(task)
  return op?.studio_path || audioContract.value.generic_studio_path
}

/** Studio-persistence defaults bucket for a task's serving operation. */
export function audioTaskDefaultsKey(task) {
  const op = audioOperationForTask(task)
  if (op?.request_defaults_key) return op.request_defaults_key
  const contract = audioContract.value
  return (
    (contract.defaults_keys || {})[op?.id || 'tasks_run'] || 'task_defaults'
  )
}

/** True when a task runs through the generic ``tasks/run`` envelope. */
export function audioTaskIsGeneric(task) {
  return audioOperationForTask(task)?.id === 'tasks_run'
}

/** Request-body envelope: openai-json | multipart-form | tasks-run. */
export function audioTaskBodyShape(task) {
  return audioOperationForTask(task)?.body_shape || 'tasks-run'
}

/** True when a contract endpoint string is the generic tasks/run surface. */
export function audioEndpointIsGeneric(endpoint) {
  const value = String(endpoint || '').trim()
  if (!value) return false
  const contract = audioContract.value
  return (
    value === (contract.generic_studio_path || OFFLINE_AUDIO_CONTRACT.generic_studio_path)
    || value === (contract.generic_native_path || OFFLINE_AUDIO_CONTRACT.generic_native_path)
  )
}

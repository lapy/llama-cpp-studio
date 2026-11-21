import { ref } from 'vue'
import { sessionFetch } from '@/api/sessionFetch.js'

/**
 * Single client-side view of the composed audio.cpp integration contract
 * (GET /api/audio-cpp/contract). Task→operation routing, public endpoint and
 * defaults-key tables come from here — never re-derived from task/family
 * strings in the frontend.
 */

export const AUDIO_CONTRACT_SCHEMA = 'audio-cpp.studio-contract/v0'
export const AUDIO_CONTRACT_PATH = '/api/audio-cpp/contract'

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

/** Fresh mutable copy of the offline floor (refs must never share tables). */
function offlineContract() {
  return {
    ...OFFLINE_AUDIO_CONTRACT,
    defaults_keys: { ...OFFLINE_AUDIO_CONTRACT.defaults_keys },
    task_routing: {},
    operations: [],
    family_routing: {},
    drift_warnings: [],
  }
}

export const audioContract = ref(offlineContract())
export const audioContractLoaded = ref(false)
export const audioContractError = ref('')
/** HTTP status of the last contract load (0 = transport/network failure). */
export const audioContractStatus = ref(0)
/** True when a 200 response carried a schema this client does not implement. */
export const audioContractSchemaMismatch = ref(false)
/** Schema string the server actually sent, for the mismatch message. */
export const audioContractSchema = ref('')

let inflight = null
let requestSequence = 0

/** Test/support hook: drop every module-level singleton back to the floor. */
export function resetAudioContract() {
  requestSequence += 1
  inflight = null
  audioContract.value = offlineContract()
  audioContractLoaded.value = false
  audioContractError.value = ''
  audioContractStatus.value = 0
  audioContractSchemaMismatch.value = false
  audioContractSchema.value = ''
}

/**
 * Fetch (or reuse) the composed contract for the active engine version.
 *
 * A forced refetch intentionally resets the composed tables before the network
 * round-trip: after an engine/model change a previous version's task_routing
 * must never answer, even if the new request fails. A failed load also leaves
 * the offline floor (never the previous payload) behind.
 */
export async function fetchAudioContract({ force = false } = {}) {
  if (inflight && !force) return inflight

  if (force) {
    audioContract.value = offlineContract()
    audioContractLoaded.value = false
    audioContractError.value = ''
    audioContractStatus.value = 0
    audioContractSchemaMismatch.value = false
    audioContractSchema.value = ''
  }

  const sequence = ++requestSequence
  const request = (async () => {
    try {
      const response = await sessionFetch(AUDIO_CONTRACT_PATH, {
        headers: { Authorization: 'Bearer local' },
      })
      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        const error = new Error(payload?.detail || `HTTP ${response.status}`)
        error.status = response.status
        throw error
      }
      const data = await response.json()
      const schema = typeof data?.schema === 'string' ? data.schema : ''
      if (schema !== AUDIO_CONTRACT_SCHEMA) {
        if (sequence === requestSequence) {
          audioContract.value = offlineContract()
          audioContractLoaded.value = false
          audioContractStatus.value = response.status || 200
          audioContractSchemaMismatch.value = true
          audioContractSchema.value = schema
          audioContractError.value =
            `Audio contract schema mismatch: expected ${AUDIO_CONTRACT_SCHEMA}, ` +
            `received ${schema || '(none)'}.`
        }
        return audioContract.value
      }
      if (sequence !== requestSequence) return audioContract.value
      audioContract.value = { ...offlineContract(), ...data, schema }
      audioContractLoaded.value = true
      audioContractError.value = ''
      audioContractStatus.value = response.status || 200
      audioContractSchemaMismatch.value = false
      audioContractSchema.value = schema
      return audioContract.value
    } catch (error) {
      // 409 = no active modern engine yet; network/5xx = capability discovery
      // failed. Both keep the offline floor so forms can still render
      // defaults, but the status is preserved so callers can say which.
      if (sequence === requestSequence) {
        audioContract.value = offlineContract()
        audioContractLoaded.value = false
        audioContractError.value = error?.message || String(error)
        audioContractStatus.value = Number(error?.status) || 0
        audioContractSchemaMismatch.value = false
        audioContractSchema.value = ''
      }
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
  const key = String(task || '')
    .trim()
    .toLowerCase()
  const byId = operationsBy()
  const opId = key ? (contract.task_routing || {})[key] : null
  let operation = opId ? byId.get(opId) : null
  if (!operation && key) {
    // ``task_routing`` is the primary table, but operations still carry their
    // own ``serves_tasks`` claims (composer merges both); honor those if a
    // payload ships a partial routing map.
    operation =
      (contract.operations || []).find((candidate) =>
        (candidate.serves_tasks || []).some(
          (servedTask) => String(servedTask).trim().toLowerCase() === key,
        ),
      ) || null
  }
  return operation || byId.get('tasks_run') || null
}

/** Public (Studio proxy) endpoint a task's requests go to. */
export function audioTaskEndpoint(task) {
  const op = audioOperationForTask(task)
  return op?.studio_path || audioContract.value.generic_studio_path
}

/**
 * Same-origin management path for a contract ``studio_path``.
 *
 * Contract paths describe the Studio proxy dataplane, where ``/audioapi`` is
 * stripped before forwarding (backend/proxy/studio/dataplane.py). The browser
 * talks to the management origin instead: backend/main.py mounts the generic
 * task route as ``/tasks/run`` under ``/v1`` and the audio routes under
 * ``/v1/audio``. Only the dataplane-only ``/audioapi`` prefix is rewritten;
 * every other contract path is already a management path.
 */
export function managementAudioPath(endpoint) {
  const value = String(endpoint || '').trim()
  if (value.startsWith('/audioapi/')) return value.slice('/audioapi'.length)
  return value
}

/** Browser-safe management path a task's requests go to. */
export function audioTaskRequestPath(task) {
  return managementAudioPath(audioTaskEndpoint(task))
}

/** Studio-persistence defaults bucket for a task's serving operation. */
export function audioTaskDefaultsKey(task) {
  const op = audioOperationForTask(task)
  if (op?.request_defaults_key) return op.request_defaults_key
  const contract = audioContract.value
  return (contract.defaults_keys || {})[op?.id || 'tasks_run'] || 'task_defaults'
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
    value === (contract.generic_studio_path || OFFLINE_AUDIO_CONTRACT.generic_studio_path) ||
    value === (contract.generic_native_path || OFFLINE_AUDIO_CONTRACT.generic_native_path)
  )
}

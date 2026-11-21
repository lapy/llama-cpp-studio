/**
 * Real-shape fixtures for the two payloads the audio workspace composes from:
 * the composed audio contract endpoint and the parameter registry.
 *
 * The operation rows mirror the backend Operation.as_dict() field-for-field
 * and the envelope mirrors the contract route in the version endpoint. Every
 * fact here is data, so a route/body mistake in the frontend cannot be hidden
 * by a mock-shaped object.
 */

import { AUDIO_CONTRACT_SCHEMA } from '@/composables/useAudioContract'

export { AUDIO_CONTRACT_SCHEMA }
export const AUDIO_CONTRACT_FINGERPRINT = 'f'.repeat(64)

function operation({
  id,
  nativePath,
  studioPath,
  method,
  encodings,
  responses,
  errors,
  servesTasks,
  upstreamOnly = false,
  requestDefaultsKey,
  bodyShape,
}) {
  return {
    id,
    native_path: nativePath,
    studio_path: studioPath,
    method,
    encodings: [...encodings],
    responses: [...responses],
    errors,
    serves_tasks: [...servesTasks],
    provenance: 'studio',
    upstream_only: upstreamOnly,
    request_defaults_key: requestDefaultsKey,
    body_shape: bodyShape,
  }
}

/** Registry order and values from the v0.9.1+ studio operation table. */
export const AUDIO_CONTRACT_OPERATIONS = Object.freeze([
  operation({
    id: 'speech',
    nativePath: '/v1/audio/speech',
    studioPath: '/v1/audio/speech',
    method: 'POST',
    encodings: ['json'],
    responses: ['audio', 'json', 'sse-events'],
    errors: 'openai-envelope',
    servesTasks: ['tts', 'vdes', 'clon'],
    requestDefaultsKey: 'speech_defaults',
    bodyShape: 'openai-json',
  }),
  operation({
    id: 'transcriptions',
    nativePath: '/v1/audio/transcriptions',
    studioPath: '/v1/audio/transcriptions',
    method: 'POST',
    encodings: ['json', 'multipart'],
    responses: ['json'],
    errors: 'openai-envelope',
    servesTasks: ['asr', 'stt'],
    requestDefaultsKey: 'transcription_defaults',
    bodyShape: 'multipart-form',
  }),
  operation({
    id: 'transcriptions_details',
    nativePath: '/v1/audio/transcriptions/details',
    studioPath: '/v1/audio/transcriptions/details',
    method: 'POST',
    encodings: ['json', 'multipart'],
    responses: ['json'],
    errors: 'openai-envelope',
    servesTasks: ['transcribe_details'],
    upstreamOnly: true,
    requestDefaultsKey: 'transcription_defaults',
    bodyShape: 'multipart-form',
  }),
  operation({
    id: 'alignments',
    nativePath: '/v1/audio/alignments',
    studioPath: '/v1/audio/alignments',
    method: 'POST',
    encodings: ['json', 'multipart'],
    responses: ['json'],
    errors: 'openai-envelope',
    servesTasks: ['align', 'alignment'],
    upstreamOnly: true,
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'multipart-form',
  }),
  operation({
    id: 'batch_transcriptions',
    nativePath: '/v1/batches/transcriptions',
    studioPath: '/v1/batches/transcriptions',
    method: 'POST',
    encodings: ['multipart', 'json'],
    responses: ['json', 'sse-events'],
    errors: 'openai-envelope',
    servesTasks: ['batch_transcribe'],
    upstreamOnly: true,
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'multipart-form',
  }),
  operation({
    id: 'voices',
    nativePath: '/v1/audio/voices',
    studioPath: '/v1/audio/voices',
    method: 'GET',
    encodings: [],
    responses: ['json'],
    errors: 'openai-envelope',
    servesTasks: [],
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'openai-json',
  }),
  operation({
    id: 'models',
    nativePath: '/v1/models',
    studioPath: '/v1/models',
    method: 'GET',
    encodings: [],
    responses: ['json'],
    errors: 'openai-envelope',
    servesTasks: [],
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'openai-json',
  }),
  operation({
    id: 'health',
    nativePath: '/health',
    studioPath: '/health',
    method: 'GET',
    encodings: [],
    responses: ['json'],
    errors: 'plain',
    servesTasks: [],
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'openai-json',
  }),
  operation({
    id: 'tasks_run',
    nativePath: '/v1/tasks/run',
    studioPath: '/audioapi/v1/tasks/run',
    method: 'POST',
    encodings: ['json', 'multipart'],
    responses: ['json', 'audio', 'sse-events'],
    errors: 'audio-cpp-error',
    servesTasks: [
      'vc',
      'svc',
      's2s',
      'vad',
      'diar',
      'sep',
      'gen',
      'spk',
      'midi',
      'turn',
      'wake',
      'cls',
    ],
    requestDefaultsKey: 'task_defaults',
    bodyShape: 'tasks-run',
  }),
])

export function audioTaskRoutingForOperations(operations) {
  const routing = {}
  for (const op of operations) {
    for (const task of op.serves_tasks || []) {
      if (!(task in routing)) routing[task] = op.id
    }
  }
  return routing
}

function familyRoutingEntry(operations, { tasks, modes, endpoints, instructionsPolicy }) {
  const entry = {
    tasks: [...tasks],
    modes: [...modes],
    endpoints: [...endpoints],
    api_endpoint: endpoints[0] || null,
    instructions_policy: instructionsPolicy || null,
  }
  if (!endpoints.length) return entry
  const operation = operations.find((op) => op.studio_path === endpoints[0])
  if (!operation) return entry
  return {
    ...entry,
    operation_id: operation.id,
    studio_path: operation.studio_path,
    request_defaults_key: operation.request_defaults_key,
  }
}

export function audioFamilyRoutingFixture(operations = AUDIO_CONTRACT_OPERATIONS) {
  return {
    omnivoice: familyRoutingEntry(operations, {
      tasks: ['tts', 'vdes', 'clon'],
      modes: ['offline', 'streaming'],
      endpoints: ['/v1/audio/speech'],
      instructionsPolicy: 'soft_tags',
    }),
    qwen3_asr: familyRoutingEntry(operations, {
      tasks: ['asr', 'stt'],
      modes: ['offline'],
      endpoints: ['/v1/audio/transcriptions'],
      instructionsPolicy: 'none',
    }),
    pocket_tts: familyRoutingEntry(operations, {
      tasks: ['tts'],
      modes: ['offline', 'streaming'],
      endpoints: ['/v1/audio/speech'],
      instructionsPolicy: 'none',
    }),
    ace_step: familyRoutingEntry(operations, {
      tasks: ['gen'],
      modes: ['offline'],
      endpoints: [],
      instructionsPolicy: 'none',
    }),
    heartmula: familyRoutingEntry(operations, {
      tasks: ['gen'],
      modes: ['offline'],
      endpoints: [],
      instructionsPolicy: 'none',
    }),
    htdemucs: familyRoutingEntry(operations, {
      tasks: ['sep'],
      modes: ['offline'],
      endpoints: [],
      instructionsPolicy: 'none',
    }),
  }
}

/**
 * Composed contract as returned by the contract route.
 *
 * `validated_through` / `engine_supported` are optional upstream-side fields
 * from the compatibility workstream; they are only included when a test asks
 * for them, exactly like the backend omits them until deployed.
 */
export function audioContractFixture({
  operations = AUDIO_CONTRACT_OPERATIONS,
  familyRouting,
  taskRouting,
  ...overrides
} = {}) {
  const clonedOperations = operations.map((op) => ({
    ...op,
    encodings: [...(op.encodings || [])],
    responses: [...(op.responses || [])],
    serves_tasks: [...(op.serves_tasks || [])],
  }))
  const routing = taskRouting || audioTaskRoutingForOperations(clonedOperations)
  return {
    schema: AUDIO_CONTRACT_SCHEMA,
    operations: clonedOperations,
    provenance: Object.fromEntries(
      clonedOperations.map((op) => [op.id, op.provenance || 'studio']),
    ),
    defaults_keys: {
      speech: 'speech_defaults',
      transcriptions: 'transcription_defaults',
      transcriptions_details: 'transcription_defaults',
      alignments: 'task_defaults',
      batch_transcriptions: 'task_defaults',
      tasks_run: 'task_defaults',
    },
    task_routing: routing,
    generic_studio_path: '/audioapi/v1/tasks/run',
    families: ['omnivoice', 'qwen3_asr', 'pocket_tts', 'ace_step', 'heartmula', 'htdemucs'],
    family_routing: familyRouting || audioFamilyRoutingFixture(clonedOperations),
    drift_warnings: [],
    upstream_contract: false,
    descriptor_fingerprint: AUDIO_CONTRACT_FINGERPRINT,
    engine_version: 'v0.9.5',
    contract_fingerprint: AUDIO_CONTRACT_FINGERPRINT,
    contract_changed: false,
    scanned_at: '2026-11-20T12:00:00Z',
    ...overrides,
  }
}

/** Minimal real-shape GET /api/models/param-registry payload. */
export function audioParamRegistryFixture(overrides = {}) {
  return {
    sections: [],
    runtime_profiles: {},
    scan_error: null,
    scan_pending: false,
    profile_fingerprint: null,
    inspection: null,
    compatibility_warnings: [],
    task_profile: null,
    request_field_groups: [],
    request_defaults_key: '',
    api_endpoint: '',
    request_policy: null,
    api_example_hint: '',
    instructions_policy: '',
    instructions_policy_source: '',
    instructions_vocabulary: null,
    supports_voice_presets: false,
    policy_family: '',
    policy_task: '',
    contract_fingerprint: '',
    contract_changed: false,
    contract_review_required: false,
    discovery_source: '',
    catalog_source: '',
    contract_drift_warnings: [],
    last_reviewed_fingerprint: '',
    sidecar_session_fields: [],
    family_dependencies: {},
    packaged_voices: [],
    workspace_request_fields: [],
    ...overrides,
  }
}

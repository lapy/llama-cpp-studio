/**
 * Reusable response payloads for the browser workspace fixture.
 *
 * This module is deliberately free of Playwright and Node-only APIs so the
 * route handler (`workspace.mjs`) and the Vitest contract check
 * (`src/api/contractFixtures.test.js`) import exactly the same data.
 */

/** Engine id shared by the audio fixture rows and the descriptor payload. */
export const AUDIO_ENGINE_ID = 'audio_cpp'
/** Management-origin contract route the Audio workspace fetches. */
export const AUDIO_CONTRACT_PATH = '/api/audio-cpp/contract'

/** One proxy request row as rendered by the Requests tab and its details dialog. */
export function proxyRequestRows({ now = Date.now() / 1000 } = {}) {
  return [
    {
      id: 'request-1',
      started_at: now - 30,
      completed_at: now - 29,
      model: 'demo-model',
      method: 'POST',
      path: '/v1/chat/completions',
      state: 'completed',
      status: 200,
      latency_ms: 150,
      input_tokens: 10,
      output_tokens: 20,
      usage_source: 'reported',
      key_name: 'Production',
    },
  ]
}

/** `/api/proxy` overview payload, including the current request ledger. */
export function proxySnapshot({ now } = {}) {
  return {
    settings: { backend: 'studio', allow_anonymous: false, proxy_port: 2000 },
    configured_backend: 'studio',
    active_backend: 'studio',
    running: [
      {
        model: 'demo-model',
        engine: 'llama_cpp',
        state: 'ready',
        ready: true,
        pid: 42,
        port: 3000,
        inflight: 0,
        request_count: 1,
        client_error_count: 0,
        server_error_count: 0,
      },
    ],
    catalog: [{ id: 'demo-model', studio_model_id: 'demo-model' }],
    requests: proxyRequestRows({ now }),
    request_total: 1,
    summary: {
      requests: 1,
      client_errors: 0,
      server_errors: 0,
      input_tokens: 10,
      output_tokens: 20,
      unknown_usage: 0,
    },
    activity: [],
  }
}

/** `/api/proxy/requests` page payload. */
export function proxyRequests({ now } = {}) {
  return { requests: proxyRequestRows({ now }), total: 1 }
}

/** Seed state for the Access keys tab. */
export function inferenceKeys() {
  return [
    {
      id: 'production',
      name: 'Production',
      prefix: 'sk-studio-preview',
      scopes: ['inference', 'models:read'],
      models: [],
    },
  ]
}

/** Append the fields the backend adds when a key is created. */
export function inferenceKeyFromRequest(body, { id = 'new-key', prefix = 'sk-studio-copy' } = {}) {
  return { ...body, id, prefix }
}

/** `/api/inference/keys` list payload. */
export function inferenceKeysResponse(keys) {
  return { keys }
}

/** `/api/inference/keys` create payload; `key` is the one-time secret. */
export function createdInferenceKey(key) {
  return { key }
}

/** `/api/access` gate state. */
export function accessState({ remote = false } = {}) {
  return { mode: remote ? 'remote' : 'local', authenticated: !remote }
}

/** `/api/status` health observation payload. */
export function workspaceStatus() {
  return {
    proxy_status: {
      backend: 'studio',
      healthy: true,
      health_observed_at: new Date().toISOString(),
    },
    persistence: {},
    runtime_observation: { quality: 'verified', observed_at: new Date().toISOString() },
  }
}

/** `/api/engines` descriptor list; the Audio workspace gates on `enabled`. */
export function engineDescriptors() {
  return {
    schema_version: 1,
    engines: [
      {
        id: AUDIO_ENGINE_ID,
        label: 'audio.cpp',
        enabled: true,
        runnable: true,
        active_version: 'v0.9.5',
        installed_versions: 1,
      },
    ],
  }
}

/** `/api/models` grouped catalog used by the model list and pickers. */
export function modelCatalog() {
  return [
    {
      base_model_name: 'Demo',
      huggingface_id: 'org/demo',
      quantizations: [
        {
          id: 'demo-model',
          name: 'Demo',
          quantization: 'Q4_K_M',
          format: 'gguf',
          is_active: true,
          run_state: 'running',
          status: 'ready',
          runtime_quality: 'verified',
          config: { engine: 'llama_cpp' },
          proxy_name: 'demo-model',
        },
      ],
    },
    {
      base_model_name: 'Speech demo',
      huggingface_id: 'org/speech',
      quantizations: [
        {
          id: 'speech-model',
          name: 'Speech demo',
          display_name: 'Speech demo',
          format: AUDIO_ENGINE_ID,
          engine: AUDIO_ENGINE_ID,
          is_active: true,
          runtime_quality: 'verified',
          config: { engine: AUDIO_ENGINE_ID },
          proxy_name: 'speech-model',
        },
      ],
    },
  ]
}

/** `/api/models/param-registry` payload; the UI tolerates an empty registry. */
export function paramRegistry(sections = []) {
  return { sections }
}

/**
 * `/api/audio-cpp/contract` composed descriptor.
 *
 * The speech fixture routes `tts` through the speech operation, so the Audio
 * workspace renders its real Speech controls instead of an offline/error
 * banner. Drift is intentionally empty for the browser journeys.
 */
export function audioContract() {
  return {
    schema: 'audio-cpp.studio-contract/v0',
    engine_version: 'v0.9.5',
    generic_studio_path: '/audioapi/v1/tasks/run',
    defaults_keys: {
      speech: 'speech_defaults',
      transcriptions: 'transcription_defaults',
      tasks_run: 'task_defaults',
    },
    task_routing: {
      tts: 'speech',
      vdes: 'speech',
      clon: 'speech',
      asr: 'transcriptions',
      stt: 'transcriptions',
      gen: 'tasks_run',
    },
    operations: [
      {
        id: 'speech',
        studio_path: '/v1/audio/speech',
        request_defaults_key: 'speech_defaults',
        body_shape: 'openai-json',
        serves_tasks: ['tts', 'vdes', 'clon'],
      },
      {
        id: 'transcriptions',
        studio_path: '/v1/audio/transcriptions',
        request_defaults_key: 'transcription_defaults',
        body_shape: 'multipart-form',
        serves_tasks: ['asr', 'stt'],
      },
      {
        id: 'tasks_run',
        studio_path: '/audioapi/v1/tasks/run',
        request_defaults_key: 'task_defaults',
        body_shape: 'tasks-run',
        serves_tasks: ['gen'],
      },
    ],
    family_routing: {},
    drift_warnings: [],
  }
}

/** `/api/models/{id}/reference-audio` payload. */
export function referenceAudio(items = []) {
  return items
}

/** `/api/models/speech-model/config` audio.cpp configuration. */
export function audioModelConfig() {
  return {
    engine: AUDIO_ENGINE_ID,
    family: 'pocket-tts',
    task: 'tts',
    model_alias: 'speech-model',
    speech_defaults: {},
  }
}

/** `/api/models/demo-model/config` llama.cpp configuration. */
export function llamaModelConfig() {
  return { engine: 'llama_cpp', engines: { llama_cpp: { ctx_size: 4096 } } }
}

/** `/api/runtime/stale` and `/api/runtime/pending` launch-plan state. */
export function runtimePlanState() {
  return { applicable: false, pending: false, stale: false }
}

/** `/api/settings/inference` payload. */
export function inferenceSettings() {
  return { public_inference_url: '' }
}

/**
 * `/api/config-history` entries.
 *
 * The backend route is `response_model=list[HistoryEntryResponse]` and the
 * frontend client types the response as an array; the fixture used to serve
 * `{ entries: [] }`, which contradicted both. No browser journey consumed it,
 * so the mock now returns the contracted array.
 */
export function configHistory(entries = []) {
  return entries
}

/** `/api/models/safetensors` payload. */
export function safetensorsModels(models = []) {
  return models
}

/** Native-origin `/v1/...` voice discovery payload for the Audio workspace. */
export function audioVoices(voices = []) {
  return { voices }
}

/** Remote-mode `/api/session` rejection body. */
export function invalidSession() {
  return { detail: 'Invalid password' }
}

/**
 * Reusable response payloads for the browser workspace fixture.
 *
 * This module is deliberately free of Playwright and Node-only APIs so the
 * route handler (`workspace.mjs`) and the Vitest contract check
 * (`src/api/contractFixtures.test.js`) import exactly the same data.
 */

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
          format: 'audio_cpp',
          engine: 'audio_cpp',
          is_active: true,
          runtime_quality: 'verified',
          config: { engine: 'audio_cpp' },
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

/** `/api/models/{id}/reference-audio` payload. */
export function referenceAudio(items = []) {
  return items
}

/** `/api/models/speech-model/config` audio.cpp configuration. */
export function audioModelConfig() {
  return {
    engine: 'audio_cpp',
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

/**
 * Client-facing inference URLs and bounded request examples.
 *
 * The management page scheme is not the proxy scheme. A local llama-swap
 * port is HTTP even when Studio itself is served over HTTPS. A configured
 * public URL replaces that local default.
 */

const AUDIO_TASKS = new Set(['asr', 'tts', 'vc', 'svc', 's2s', 'gen', 'clon'])

export function normalizePublicInferenceUrl(value) {
  const text = String(value || '').trim().replace(/\/+$/, '')
  if (!text) return ''
  let parsed
  try {
    parsed = new URL(text)
  } catch {
    return ''
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return ''
  if (parsed.username || parsed.password) return ''
  return `${parsed.protocol}//${parsed.host}${parsed.pathname.replace(/\/+$/, '')}`
}

export function inferenceOrigin({ publicUrl = '', proxyPort = 2000 } = {}) {
  const configured = normalizePublicInferenceUrl(publicUrl)
  if (configured) return configured
  const port = Number(proxyPort)
  const resolved = Number.isFinite(port) && port > 0 ? port : 2000
  return `http://127.0.0.1:${resolved}`
}

export function connectKind(model) {
  if (!model) return 'chat'
  const engine = model.config?.engine || model.engine || ''
  const tasks = model.tasks || model.config?.tasks || []
  const task = String(model.config?.task || model.task || '').toLowerCase()
  if (engine === 'audio_cpp' || model.format === 'audio_cpp' || AUDIO_TASKS.has(task) || tasks.some((item) => AUDIO_TASKS.has(item))) {
    return 'audio'
  }
  if (model.is_embedding_model || tasks.includes('embeddings') || model.pipeline_tag === 'feature-extraction') {
    return 'embeddings'
  }
  return 'chat'
}

export function connectRequest(model) {
  const id = model?.llama_swap_id || model?.proxy_name || model?.config?.model_alias || model?.id || ''
  const kind = connectKind(model)
  if (kind === 'embeddings') {
    return {
      kind,
      path: '/v1/embeddings',
      body: { model: id, input: 'ping' },
    }
  }
  if (kind === 'audio') {
    return {
      kind,
      path: '/audioapi/v1/tasks/run',
      body: { model: id, input: '' },
    }
  }
  return {
    kind,
    path: '/v1/chat/completions',
    body: {
      model: id,
      messages: [{ role: 'user', content: 'Reply with the word pong.' }],
      max_tokens: 16,
    },
  }
}

export function connectEndpoint(model, options = {}) {
  const request = connectRequest(model)
  return `${inferenceOrigin(options)}${request.path}`
}

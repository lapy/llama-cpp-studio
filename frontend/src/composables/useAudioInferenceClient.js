import { sessionFetch } from '@/api/sessionFetch.js'
import { audioTaskBodyShape, audioTaskRequestPath } from '@/composables/useAudioContract.js'

/**
 * Audio inference through the Studio origin. Studio handles transport and
 * format compatibility; audio.cpp owns request semantics and model behavior.
 *
 * Route/endpoint/defaults-key facts come from the composed integration
 * contract (``useAudioContract``); this module only moves bytes and builds the
 * body shape the contract assigns to the serving operation. The only literal
 * kept here is the UI task→tab presentation map at the bottom, which is not a
 * transport route.
 */

const SPEECH_RATE_KEYS = ['speed', 'speaking_rate']

/** Absolute Studio-origin URL for a browser-safe management path. */
function studioRequestUrl(path) {
  const value = String(path || '')
  if (/^https?:\/\//i.test(value)) return value
  const origin =
    typeof window !== 'undefined' && window.location?.origin ? window.location.origin : ''
  return `${origin}${value.startsWith('/') ? value : `/${value}`}`
}

/**
 * Resolve the browser/management path for a task and require the body shape
 * this caller knows how to produce. With no contract loaded the offline floor
 * says ``tasks-run``, so speech/transcription callers fail loudly instead of
 * posting a body the generic route would misinterpret.
 */
function contractEndpointFor(task, bodyShape, action) {
  const requestPath = audioTaskRequestPath(task)
  const actual = audioTaskBodyShape(task)
  if (actual !== bodyShape) {
    throw new Error(
      `${action} needs the audio.cpp integration contract ` +
        `(expected ${bodyShape}, got ${actual} from ${requestPath}). Reload capabilities.`,
    )
  }
  return requestPath
}

/** Request-option keys the model scan advertised as speed controls. */
export function acceptedSpeechRateKeys(registry) {
  const keys = []
  const sections = Array.isArray(registry?.sections) ? registry.sections : []
  for (const section of sections) {
    const params = Array.isArray(section?.params) ? section.params : []
    for (const param of params) {
      if (param?.scope !== 'request_option') continue
      const key = String(param?.key || '')
      if (SPEECH_RATE_KEYS.includes(key) && !keys.includes(key)) keys.push(key)
    }
  }
  return keys
}

/**
 * Top-level speech fields for rates the scan accepts.
 * audio.cpp reads ``speed`` / ``speaking_rate`` on the speech JSON body and
 * rejects them when the model has no speed control.
 */
export function speechRateRequestFields(defaults, acceptedKeys) {
  const accepted = new Set(acceptedKeys || [])
  const options =
    defaults?.options && typeof defaults.options === 'object' && !Array.isArray(defaults.options)
      ? defaults.options
      : {}
  const fields = {}
  for (const key of SPEECH_RATE_KEYS) {
    if (!accepted.has(key)) continue
    const raw = options[key] != null && options[key] !== '' ? options[key] : defaults?.[key]
    if (raw == null || raw === '' || typeof raw === 'boolean') continue
    const value = typeof raw === 'number' ? raw : Number(raw)
    if (!Number.isFinite(value) || value <= 0) continue
    fields[key] = value
  }
  return fields
}

export function audioInferenceModelId(model, config = null) {
  const cfg = config || model?.config || {}
  return cfg.model_alias || model?.proxy_model_id || model?.proxy_name || model?.id || ''
}

export function studioAudioBaseUrl() {
  if (typeof window !== 'undefined' && window.location?.origin) {
    return `${window.location.origin}/v1/audio`
  }
  return '/v1/audio'
}

/**
 * Normalize Studio/workspace bodies to audio.cpp ``request`` object fields.
 */
export function tasksRunRequestObject(input = {}) {
  let requestObj = input
  if (requestObj && typeof requestObj === 'object' && !Array.isArray(requestObj)) {
    if (requestObj.request && typeof requestObj.request === 'object') {
      requestObj = requestObj.request
    } else if (requestObj.input && typeof requestObj.input === 'object') {
      requestObj = requestObj.input
    }
  }
  if (!requestObj || typeof requestObj !== 'object' || Array.isArray(requestObj)) {
    return {}
  }
  const normalized = {}
  for (const [key, value] of Object.entries(requestObj)) {
    if (value == null || value === '') continue
    if (['model', 'task', 'input', 'request', 'busy_timeout_ms'].includes(key)) continue
    normalized[key] = value
  }
  return normalized
}

async function readErrorDetail(response) {
  const text = await response.text().catch(() => '')
  if (!text) return `HTTP ${response.status}`
  try {
    const json = JSON.parse(text)
    if (typeof json.detail === 'string') return json.detail
    if (json.error?.message) return json.error.message
    if (json.message) return json.message
    return text
  } catch {
    return text
  }
}

/**
 * POST the contract's speech route — returns { blob, contentType }
 */
export async function synthesizeSpeech({
  task = 'tts',
  modelId,
  input,
  voice,
  extras = {},
  signal,
} = {}) {
  const body = {
    model: modelId,
    input,
    ...extras,
  }
  if (voice != null && voice !== '') body.voice = voice
  const endpoint = contractEndpointFor(task, 'openai-json', 'Speech synthesis')

  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer local',
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || 'audio/wav'
  const blob = await response.blob()
  return { blob, contentType }
}

/**
 * Read one audio.cpp speech SSE body (speech.audio.delta / speech.text.delta /
 * speech.audio.done / [DONE]). Handlers fire as frames arrive; resolves with
 * the concatenated PCM16LE bytes plus text and timing.
 */
async function readSpeechSseStream(response, { onAudio, onText, onTiming } = {}) {
  const chunks = []
  let pcmBytes = 0
  let text = ''
  let timing = null
  let streamError = ''
  let buffer = ''

  const decodeBase64 = (value) => {
    const binary = atob(value)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index)
    return bytes
  }

  const handleEvent = (payload) => {
    let event
    try {
      event = JSON.parse(payload)
    } catch {
      return
    }
    if (event.type === 'speech.audio.delta' && typeof event.audio === 'string') {
      const bytes = decodeBase64(event.audio)
      if (bytes.length) {
        chunks.push(bytes)
        pcmBytes += bytes.length
        onAudio?.(bytes, chunks.length - 1)
      }
    } else if (event.type === 'speech.text.delta' && typeof event.delta === 'string') {
      text += event.delta
      onText?.(event.delta)
    } else if (event.type === 'speech.text.done' && typeof event.text === 'string') {
      text = event.text
    } else if (event.type === 'speech.audio.done') {
      timing = event.timing || null
      if (timing) onTiming?.(timing)
    } else if (event.type === 'error') {
      streamError = event.message || event.error?.message || 'Speech stream failed'
    }
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let ended = false
  while (!ended) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE frames are separated by a blank line; the engine writes "data: <json>\n\n".
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1 && !ended) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      for (const line of frame.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data:')) continue
        const payload = trimmed.slice(5).trim()
        if (!payload) continue
        if (payload === '[DONE]') {
          ended = true
          break
        }
        handleEvent(payload)
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
  if (streamError) throw new Error(streamError)

  const pcm = new Uint8Array(pcmBytes)
  let offset = 0
  for (const chunk of chunks) {
    pcm.set(chunk, offset)
    offset += chunk.length
  }
  return { pcm, text, timing }
}

/**
 * Native PCM speech streaming (audio.cpp mode=streaming).
 *
 * Sends `stream: true, stream_format: 'sse', response_format: 'pcm'` and reads
 * the OpenAI-shape SSE deltas chunk-by-chunk with no buffering anywhere on the
 * path. SSE is deliberate: it is the only response shape Studio proxy's reverse
 * proxy flushes per event.
 *
 * onAudio(pcmBytes, frame) fires per delta so callers can play immediately;
 * onText(textDelta) fires when the model writes text. Resolves with the
 * concatenated PCM16LE bytes, { timing, text } (timing carries ttft_ms), and a
 * wav Blob assembled from the same buffers.
 */
export async function streamSpeech({
  task = 'tts',
  modelId,
  input,
  voice,
  voiceRef,
  referenceText,
  sampleRate = 24000,
  channels = 1,
  returnText = false,
  extras = {},
  signal,
  onAudio,
  onText,
  onTiming,
} = {}) {
  const body = {
    model: modelId,
    input,
    stream: true,
    stream_format: 'sse',
    response_format: 'pcm',
    ...extras,
  }
  if (voice != null && voice !== '') body.voice = voice
  if (voiceRef != null && voiceRef !== '') body.voice_ref = voiceRef
  if (referenceText != null && referenceText !== '') body.reference_text = referenceText
  if (returnText) body.return_text = true
  const endpoint = contractEndpointFor(task, 'openai-json', 'Streaming speech')

  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      Authorization: 'Bearer local',
    },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('text/event-stream')) {
    // The engine answered non-streamed (model not running with mode=streaming,
    // or an older build without stream support). Fall back to one full blob.
    const blob = await response.blob()
    return { pcm: new Uint8Array(0), blob, text: '', timing: null, fellBack: true }
  }

  const { pcm, text, timing } = await readSpeechSseStream(response, { onAudio, onText, onTiming })
  const wavBlob = pcm.length
    ? pcmToWavBlob(pcm, { sampleRate, channels })
    : new Blob([], { type: 'audio/wav' })
  return { pcm, blob: wavBlob, text, timing, fellBack: false }
}

/** Wrap raw PCM16LE bytes in a WAV header so the result is playable/downloadable. */
export function pcmToWavBlob(pcm, { sampleRate = 24000, channels = 1, bitsPerSample = 16 } = {}) {
  const byteRate = (sampleRate * channels * bitsPerSample) / 8
  const header = new ArrayBuffer(44)
  const view = new DataView(header)
  const ascii = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index))
    }
  }
  ascii(0, 'RIFF')
  view.setUint32(4, 36 + pcm.byteLength, true)
  ascii(8, 'WAVE')
  ascii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, (channels * bitsPerSample) / 8, true)
  view.setUint16(34, bitsPerSample, true)
  ascii(36, 'data')
  view.setUint32(40, pcm.byteLength, true)
  return new Blob([header, pcm], { type: 'audio/wav' })
}

/**
 * POST the contract's transcription route (multipart) — returns parsed JSON or { text }
 */
export async function transcribeAudio({
  task = 'asr',
  modelId,
  file,
  filename,
  language,
  prompt,
  details = false,
  extras = {},
  signal,
} = {}) {
  const form = new FormData()
  form.append('model', modelId)
  form.append('file', file, filename || file.name || 'audio.wav')
  if (language) form.append('language', language)
  if (prompt) form.append('prompt', prompt)
  for (const [key, value] of Object.entries(extras)) {
    if (value == null || value === '') continue
    form.append(key, typeof value === 'string' ? value : String(value))
  }

  const servingTask = details ? 'transcribe_details' : task
  const endpoint = contractEndpointFor(servingTask, 'multipart-form', 'Transcription')
  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers: {
      Authorization: 'Bearer local',
    },
    body: form,
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return response.json()
  }
  const text = await response.text()
  return { text }
}

function batchItem(item, index) {
  if (typeof item === 'string') return { name: `File ${index + 1}`, text: item }
  if (!item || typeof item !== 'object') {
    return { name: `File ${index + 1}`, text: String(item ?? '') }
  }
  const name = item.filename || item.file || item.name || item.id || `File ${index + 1}`
  const text = item.text ?? item.transcript ?? item.transcription
  return {
    name: String(name),
    text: typeof text === 'string' ? text : JSON.stringify(item, null, 2),
  }
}

export function parseBatchTranscriptionPayload(contentType, bodyText) {
  const type = String(contentType || '').toLowerCase()
  const text = String(bodyText || '')
  const trimmed = text.trim()
  if (type.includes('application/json') || trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const payload = JSON.parse(trimmed)
      if (Array.isArray(payload)) return { results: payload.map(batchItem) }
      if (payload && typeof payload === 'object') {
        const list = payload.results || payload.transcriptions || payload.files
        if (Array.isArray(list)) return { ...payload, results: list.map(batchItem) }
        if (typeof payload.text === 'string') {
          return { text: payload.text, results: [batchItem(payload, 0)] }
        }
      }
    } catch {
      /* Fall through to the event-stream and plain-text readers. */
    }
  }
  if (type.includes('text/event-stream') || text.includes('\ndata:') || text.startsWith('data:')) {
    const events = []
    for (const line of text.split(/\r?\n/)) {
      if (!line.startsWith('data:')) continue
      const data = line.slice(5).trim()
      if (!data || data === '[DONE]') continue
      let event
      try {
        event = JSON.parse(data)
      } catch {
        event = { text: data }
      }
      // The native batch route ends with batch.transcription.done (aggregate
      // timing, no result); it is not a transcription row.
      if (typeof event?.type === 'string' && event.type.endsWith('.done')) continue
      events.push(event)
    }
    if (events.length) return { results: events.map(batchItem) }
  }
  return { text, results: text ? [{ name: 'Batch', text }] : [] }
}

/**
 * POST the contract's native batch transcription route — one multipart request,
 * up to 32 files. Returns { results: [{ name, text }], text? }.
 */
export async function transcribeAudioBatch({
  task = 'batch_transcribe',
  modelId,
  files,
  language,
  prompt,
  signal,
} = {}) {
  const list = Array.from(files || []).filter(Boolean)
  if (!list.length) throw new Error('Choose at least one audio file.')
  if (list.length > 32) throw new Error('Batch transcription accepts at most 32 files.')
  const form = new FormData()
  form.append('model', modelId)
  for (const file of list) {
    form.append('file', file, file.name || 'audio.wav')
  }
  if (language) form.append('language', language)
  if (prompt) form.append('prompt', prompt)
  const endpoint = contractEndpointFor(task, 'multipart-form', 'Batch transcription')

  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers: { Authorization: 'Bearer local' },
    body: form,
    signal,
  })
  if (!response.ok) throw new Error(await readErrorDetail(response))
  const contentType = response.headers.get('content-type') || ''
  return parseBatchTranscriptionPayload(contentType, await response.text())
}

/** Voice ids are resolved by the running engine, including its voice library. */
export async function fetchAudioVoices({ modelId, signal } = {}) {
  const response = await sessionFetch(
    `${studioAudioBaseUrl()}/voices?model=${encodeURIComponent(modelId)}`,
    { headers: { Authorization: 'Bearer local' }, signal },
  )
  if (!response.ok) throw new Error(await readErrorDetail(response))
  const result = await response.json()
  return Array.isArray(result.voices) ? result.voices.filter((id) => typeof id === 'string') : []
}

/** Forced alignment accepts a client upload and the known transcript. */
export async function alignAudio({ modelId, file, text, language, signal } = {}) {
  const form = new FormData()
  form.append('model', modelId)
  form.append('file', file, file.name || 'audio.wav')
  form.append('text', text)
  if (language) form.append('language', language)
  const endpoint = contractEndpointFor('align', 'multipart-form', 'Forced alignment')
  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers: { Authorization: 'Bearer local' },
    body: form,
    signal,
  })
  if (!response.ok) throw new Error(await readErrorDetail(response))
  return response.json()
}

function appendMultipartValue(form, key, value) {
  if (value == null || value === '') return
  if (Array.isArray(value)) {
    for (const item of value) appendMultipartValue(form, key, item)
    return
  }
  if (typeof Blob !== 'undefined' && value instanceof Blob) {
    form.append(key, value, value.name || key)
    return
  }
  form.append(key, typeof value === 'object' ? JSON.stringify(value) : String(value))
}

/**
 * POST the operation the contract routes ``task`` to. The operation's
 * ``body_shape`` decides the envelope: generic tasks keep audio.cpp's
 * ``{model, request}`` object, OpenAI-shaped operations send flat JSON, and
 * multipart operations send FormData. No task→route table lives here.
 */
export async function runAudioTask({ modelId, task, input = {}, busyTimeoutMs, signal } = {}) {
  const endpoint = audioTaskRequestPath(task)
  const shape = audioTaskBodyShape(task)
  const request = tasksRunRequestObject(input)
  let headers = { Authorization: 'Bearer local' }
  let body

  if (shape === 'multipart-form') {
    const form = new FormData()
    form.append('model', modelId)
    for (const [key, value] of Object.entries(request)) appendMultipartValue(form, key, value)
    body = form
  } else if (shape === 'openai-json') {
    headers = { ...headers, 'Content-Type': 'application/json' }
    body = JSON.stringify({ model: modelId, ...request })
  } else {
    const envelope = { model: modelId, request }
    if (busyTimeoutMs != null) envelope.busy_timeout_ms = busyTimeoutMs
    headers = { ...headers, 'Content-Type': 'application/json' }
    body = JSON.stringify(envelope)
  }

  const response = await sessionFetch(studioRequestUrl(endpoint), {
    method: 'POST',
    headers,
    body,
    signal,
  })
  if (!response.ok) {
    throw new Error(await readErrorDetail(response))
  }
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    return response.json()
  }
  return { blob: await response.blob(), contentType }
}

/**
 * UI presentation only: which workspace tab a configured task opens. Route
 * selection never reads this table; it is the last-resort offline mapping for
 * known task names when no composed contract is loaded. With a contract, the
 * serving operation id decides between Speech/Transcribe/Analyze and the
 * schema-driven generic task form.
 */
export function taskKindFromConfig(config = {}) {
  const task = String(config.task || '').toLowerCase()
  if (['asr', 'stt', 'transcription'].includes(task)) return 'transcribe'
  if (['vc', 'voice_conversion'].includes(task)) return 'convert'
  if (['svc', 's2s'].includes(task)) return 'convert'
  if (['sep', 'separation'].includes(task)) return 'separate'
  if (['vad', 'diar', 'diarization', 'align', 'alignment'].includes(task)) return 'analyze'
  if (['gen', 'music', 'sfx'].includes(task)) return 'music'
  if (['design', 'vdes', 'voice_design'].includes(task)) return 'design'
  // clon/clone stay on Speech (TTS / clone) — OpenAI speech path
  if (['tts', 'speech', 'clon', 'clone'].includes(task) || !task) return 'speech'
  return 'task'
}

/** Map workspace tab id for deep links from a model config. */
export function audioTabFromConfig(config = {}) {
  return taskKindFromConfig(config)
}

function base64ToUint8Array(b64) {
  const binary = atob(String(b64 || '').replace(/\s/g, ''))
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/**
 * Decode audio.cpp ``/v1/tasks/run`` JSON into playable WAV blobs.
 * Primary track is ``audio`` (base64 WAV); separation also returns
 * ``named_audio_outputs[{id, audio}]``.
 */
export function extractAudioClipsFromTaskResult(result) {
  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    return { clips: [], meta: null }
  }
  if (result.blob instanceof Blob) {
    return {
      clips: [{ id: 'audio', label: 'Audio', blob: result.blob, filename: 'audio.wav' }],
      meta: null,
    }
  }

  const clips = []
  if (typeof result.audio === 'string' && result.audio) {
    const bytes = base64ToUint8Array(result.audio)
    clips.push({
      id: 'audio',
      label: 'Audio',
      blob: new Blob([bytes], { type: 'audio/wav' }),
      filename: 'audio.wav',
      sample_rate: result.sample_rate,
      channels: result.channels,
    })
  }
  for (const [index, named] of (result.named_audio_outputs || []).entries()) {
    if (typeof named?.audio !== 'string' || !named.audio) continue
    const id = String(named.id || `stem-${index + 1}`)
    const bytes = base64ToUint8Array(named.audio)
    clips.push({
      id,
      label: id,
      blob: new Blob([bytes], { type: 'audio/wav' }),
      filename: `${id.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'stem'}.wav`,
      sample_rate: named.sample_rate,
      channels: named.channels,
    })
  }

  const meta = { ...result }
  delete meta.audio
  delete meta.named_audio_outputs
  return { clips, meta: Object.keys(meta).length ? meta : null }
}

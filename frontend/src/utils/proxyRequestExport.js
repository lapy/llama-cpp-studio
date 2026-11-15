/**
 * Builders for exporting one captured Studio proxy request.
 *
 * The proxy retains request metadata only: it records no request headers and
 * no request or response bodies. These builders include `headers` / `body`
 * when a captured request happens to carry them and otherwise produce the most
 * accurate command or document the metadata allows. Nothing is redacted:
 * captured data carries no secret marker, and silently dropping a captured
 * header would hide evidence the operator asked to inspect.
 *
 * Every builder is pure. `downloadTextFile` is the only browser touchpoint and
 * exists here so the inspector has a single export import.
 */

const ABSOLUTE_URL = /^[a-z][a-z0-9+.-]*:\/\//i

/** Quote one value for a POSIX-style shell, including apostrophes. */
export function shellQuote(value) {
  return `'${String(value ?? '').replaceAll("'", `'"'"'`)}'`
}

/**
 * Normalize captured request headers to ordered `{ name, value }` pairs.
 * Accepts a name/value map, `[name, value]` pairs, or HAR-style objects.
 */
export function requestHeaders(request) {
  const raw = request?.headers
  if (Array.isArray(raw)) {
    return raw
      .flatMap((entry) => {
        if (Array.isArray(entry) && entry.length >= 2) {
          return [{ name: String(entry[0]), value: String(entry[1] ?? '') }]
        }
        if (entry && typeof entry === 'object' && entry.name != null) {
          return [{ name: String(entry.name), value: String(entry.value ?? '') }]
        }
        return []
      })
      .filter((header) => header.name !== '')
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw)
      .filter(([name]) => name !== '')
      .map(([name, value]) => ({
        name,
        value: Array.isArray(value) ? value.join(', ') : String(value ?? ''),
      }))
  }
  return []
}

/** Captured request body as text, or null when the capture has none. */
export function requestBodyText(request) {
  const body = request?.body
  if (body == null || body === '') return null
  if (typeof body === 'string') return body
  const encoded = JSON.stringify(body)
  return encoded == null ? null : encoded
}

/** Whether the capture carries anything cURL could replay beyond method/URL. */
export function hasCapturedPayload(request) {
  return requestHeaders(request).length > 0 || requestBodyText(request) !== null
}

/**
 * Absolute request URL for the current origin. A captured absolute URL wins; a
 * missing or non-http(s) origin leaves the path relative instead of inventing
 * a host.
 */
export function buildRequestUrl(request, { origin = '' } = {}) {
  const raw = String(request?.path || '/').trim() || '/'
  if (ABSOLUTE_URL.test(raw)) return raw
  const path = raw.startsWith('/') ? raw : `/${raw}`
  const base = normalizeOrigin(origin)
  return base ? `${base}${path}` : path
}

/** Build a POSIX-safe curl command for the captured request. */
export function buildCurlCommand(request, { origin = '' } = {}) {
  const lines = [
    `curl --request ${requestMethod(request)} ${shellQuote(buildRequestUrl(request, { origin }))}`,
  ]
  for (const header of requestHeaders(request)) {
    lines.push(`  --header ${shellQuote(`${header.name}: ${header.value}`)}`)
  }
  const body = requestBodyText(request)
  if (body !== null) lines.push(`  --data ${shellQuote(body)}`)
  return lines.join(' \\\n')
}

/** Raw captured request object, pretty-printed for a `.json` download. */
export function buildJsonExport(request) {
  return `${JSON.stringify(request ?? {}, null, 2)}\n`
}

/** Build a single-entry HAR 1.2 document for the captured request. */
export function buildHarExport(request, { origin = '', appVersion = '1.0.0' } = {}) {
  const url = buildRequestUrl(request, { origin })
  const headers = requestHeaders(request)
  const body = requestBodyText(request)
  const duration = requestDurationMs(request)
  const responseBytes = nonNegativeNumber(request?.output_bytes)
  const entry = {
    startedDateTime: requestStartedDateTime(request),
    time: duration,
    request: {
      method: requestMethod(request),
      url,
      httpVersion: requestHttpVersion(request),
      cookies: [],
      headers,
      queryString: queryString(url),
      headersSize: -1,
      bodySize: body !== null ? byteLength(body) : nonNegativeNumber(request?.input_bytes),
      ...(body !== null ? { postData: { mimeType: contentType(headers) || '', text: body } } : {}),
    },
    response: {
      status: responseStatus(request),
      statusText: typeof request?.status_text === 'string' ? request.status_text : '',
      httpVersion: requestHttpVersion(request),
      cookies: [],
      headers: [],
      content: { size: responseBytes, mimeType: '' },
      redirectURL: '',
      headersSize: -1,
      bodySize: responseBytes,
    },
    cache: {},
    timings: { send: -1, wait: duration, receive: -1 },
  }
  return {
    log: {
      version: '1.2',
      creator: { name: 'llama-cpp-studio', version: String(appVersion || '1.0.0') },
      entries: [entry],
    },
  }
}

/** Filesystem-safe `request-<id>.json` / `request-<id>.har` download name. */
export function exportFilename(id, extension = 'json') {
  const suffix = String(extension).toLowerCase() === 'har' ? 'har' : 'json'
  const safe = String(id ?? '')
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '')
    .slice(0, 96)
  return `request-${safe || 'unknown'}.${suffix}`
}

/** Save text through a Blob object URL and revoke that URL after the click. */
export function downloadTextFile(content, filename, mimeType = 'application/json') {
  const blob = new Blob([String(content ?? '')], { type: `${mimeType};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.hidden = true
  document.body.append(anchor)
  try {
    anchor.click()
  } finally {
    anchor.remove()
    URL.revokeObjectURL(url)
  }
}

function normalizeOrigin(origin) {
  const value = String(origin || '').trim()
  if (!value) return ''
  try {
    const parsed = new URL(value)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return ''
    if (parsed.username || parsed.password) return ''
    return parsed.origin
  } catch {
    return ''
  }
}

function requestMethod(request) {
  const method = String(request?.method || 'GET')
    .trim()
    .toUpperCase()
  return method || 'GET'
}

function requestHttpVersion(request) {
  // ASGI's http_version is not retained in the captured request, but HAR
  // requires a string. The Studio proxy listener speaks HTTP/1.1.
  const captured = String(request?.http_version || '').trim()
  return captured || 'HTTP/1.1'
}

function requestStartedDateTime(request) {
  for (const value of [request?.started_at, request?.completed_at]) {
    const seconds = Number(value)
    if (Number.isFinite(seconds) && seconds >= 0) {
      return new Date(seconds * 1000).toISOString()
    }
  }
  return new Date(0).toISOString()
}

function requestDurationMs(request) {
  const latency = Number(request?.latency_ms)
  if (Number.isFinite(latency) && latency >= 0) return latency
  const started = Number(request?.started_at)
  const completed = Number(request?.completed_at)
  if (Number.isFinite(started) && Number.isFinite(completed) && completed >= started) {
    return Math.round((completed - started) * 1000)
  }
  return 0
}

function responseStatus(request) {
  const status = Number(request?.http_status ?? request?.status)
  return Number.isInteger(status) && status >= 0 && status <= 999 ? status : 0
}

function queryString(url) {
  try {
    return [...new URL(url, 'http://localhost').searchParams].map(([name, value]) => ({
      name,
      value,
    }))
  } catch {
    return []
  }
}

function contentType(headers) {
  return headers.find((header) => header.name.toLowerCase() === 'content-type')?.value || ''
}

function byteLength(value) {
  return new TextEncoder().encode(String(value)).length
}

function nonNegativeNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) && number > 0 ? Math.round(number) : 0
}

/**
 * Pure presentation helpers shared by the proxy request table, the request
 * inspector, the gateway overview, and the access-key ledger. Proxy timestamps
 * are Unix seconds; metrics are milliseconds.
 */
export function formatTime(value) {
  return value ? new Date(value * 1000).toLocaleString() : 'Unknown'
}

export function metric(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)} ms`
}

export function formatBuildIdentity(value) {
  if (!value) return '—'
  try {
    const parsed = JSON.parse(value)
    const bits = [parsed.version, parsed.source_commit && `commit ${parsed.source_commit}`].filter(
      Boolean,
    )
    if (bits.length) return bits.join(' · ')
    if (parsed.executable?.path) return parsed.executable.path
  } catch {
    /* plain identity string */
  }
  return String(value)
}

export function requestSeverity(request) {
  return (
    {
      completed: 'success',
      rejected: 'warn',
      cancelled: 'warn',
      failed: 'danger',
      streaming: 'info',
      forwarding: 'info',
    }[request.state] || 'secondary'
  )
}

/** Integer counts with an em dash for values the proxy has not reported. */
export function number(value) {
  return value == null ? '—' : Number(value).toLocaleString()
}

/** One-decimal throughput; null stays visibly unknown instead of zero. */
export function rate(value) {
  return value == null ? '—' : Number(value).toFixed(1)
}

/** A 0..1 ratio rendered as a percentage. */
export function percent(value) {
  return value == null ? '—' : `${(Number(value) * 100).toFixed(1)}%`
}

export function formatBytes(value) {
  const bytes = Number(value || 0)
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / 1048576).toFixed(1)} MiB`
}

export function shortRevision(value) {
  return String(value || 'unknown').slice(0, 12)
}

/** Request rows record Unix seconds; the table shows a wall-clock time. */
export function formatClock(value) {
  return value ? new Date(value * 1000).toLocaleTimeString() : '—'
}

/** Access-key timestamps are ISO strings, unlike proxy request timestamps. */
export function formatDate(value) {
  return new Date(value).toLocaleString()
}

export function formatAgo(iso) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  return `${Math.floor(seconds / 86400)} d ago`
}

/** A configured budget; null means the policy is unlimited. */
export function formatAllowance(value) {
  return value == null ? 'unlimited' : Number(value).toLocaleString()
}

/**
 * A buffered (non-streaming) response has no first-output event before the
 * complete body: TTFT would equal total duration, so label the transport
 * instead of presenting a fabricated token-latency number.
 */
export function ttftLabel(request) {
  if (request.ttft_ms != null) return metric(request.ttft_ms)
  if (request.response_transport === 'buffered') return 'non-stream'
  return '—'
}

export function ttftDetail(request, semantics = {}) {
  if (request.ttft_ms != null) return metricNote('ttft_ms', semantics)
  if (request.response_transport === 'buffered') {
    return request.requested_stream === true
      ? 'The client requested streaming but the engine returned a buffered JSON body; TTFT is not observable.'
      : 'The client requested a buffered (non-streaming) response; the first output event is the complete body, so TTFT is not observable.'
  }
  return 'TTFT is unavailable because the engine did not emit recognizable generated-output events.'
}

export function tokenDetail(request) {
  const bits = []
  if (request.input_tokens_total != null) bits.push(`logical input ${request.input_tokens_total}`)
  if (request.input_tokens_cached != null) bits.push(`cached ${request.input_tokens_cached}`)
  if (request.input_tokens_uncached != null) bits.push(`uncached ${request.input_tokens_uncached}`)
  if (request.input_tokens_processed != null)
    bits.push(`engine processed ${request.input_tokens_processed}`)
  if (request.output_tokens_reasoning != null)
    bits.push(`reasoning ${request.output_tokens_reasoning}`)
  if (request.usage_status) bits.push(`usage ${request.usage_status}`)
  if (request.input_tokens_provenance) bits.push(`input from ${request.input_tokens_provenance}`)
  return bits.join(' · ') || 'Usage was not reported for this request.'
}

/**
 * Tooltip text for a metric. `input_cache` has a fixed definition because the
 * proxy computes it locally; every other metric uses the semantics the proxy
 * reports with the payload, so the UI never invents a definition.
 */
export function metricNote(key, semantics = {}) {
  if (key === 'input_cache') {
    return 'Cache hit ratio = sum(cached input) / sum(total input) over requests with reported cache details. Requests with unknown cache reporting are excluded from the denominator.'
  }
  const note = semantics?.[key]
  if (!note) return ''
  return `${note.label}: ${note.definition} Origin: ${note.origin}`
}

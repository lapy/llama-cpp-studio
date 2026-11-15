/**
 * Pure presentation helpers shared by the proxy request table and the request
 * inspector. Proxy timestamps are Unix seconds; metrics are milliseconds.
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

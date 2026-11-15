/**
 * Shared timing and wording for the live-update (SSE) connection.
 *
 * The progress store owns the connection state; these helpers keep the footer,
 * the System diagnostics row, and the paused banner from drifting apart.
 */

// Reconnect backoff: 1s, 2s, 4s, ... capped at 30s. The capped exponential
// base is multiplied by a jitter factor in [0.8, 1.2] and then capped again, so
// a crowd of tabs does not retry in lockstep while the delay still never
// exceeds the documented ceiling.
export const RECONNECT_BASE_DELAY_MS = 1_000
export const RECONNECT_MAX_DELAY_MS = 30_000
export const RECONNECT_JITTER_RATIO = 0.2

// After this many failed attempts the status reads "Offline" rather than
// "Reconnecting". Automatic retries continue with the capped delay.
export const OFFLINE_AFTER_ATTEMPTS = 4

// A connected stream with in-flight work is expected to keep producing
// events. The backend heartbeat is an SSE comment (`: heartbeat` every
// HEARTBEAT_SECONDS = 15s in backend/operations/progress.py); the SSE spec and
// EventSource discard comments, so JavaScript cannot see those heartbeats. The
// store therefore only declares a stream stale while active tasks are known,
// where 45s of silence is evidence of a half-open connection rather than a
// healthy idle session. See stores/progress.js for the watchdog.
export const STALE_CONNECTION_MS = 45_000
export const STALE_CHECK_INTERVAL_MS = 5_000

// How long a known outage may last before the polite "paused" banner appears.
export const LIVE_UPDATES_PAUSED_DELAY_MS = 10_000

/**
 * Delay before retry number `attempt` (1-based). Exposed for unit tests and to
 * keep the store's scheduling policy in one place.
 */
export function reconnectDelay(attempt, random = Math.random) {
  const retry = Math.max(1, Math.trunc(Number(attempt) || 1))
  const base = Math.min(RECONNECT_MAX_DELAY_MS, RECONNECT_BASE_DELAY_MS * 2 ** (retry - 1))
  const jitter = 1 - RECONNECT_JITTER_RATIO + random() * RECONNECT_JITTER_RATIO * 2
  return Math.min(RECONNECT_MAX_DELAY_MS, Math.round(base * jitter))
}

/** Compact, human-readable age: "just now", "2s ago", "3m ago", "2h ago". */
export function formatAgeLabel(ageMs) {
  if (ageMs == null || !Number.isFinite(ageMs)) return 'never'
  const seconds = Math.max(0, Math.floor(ageMs / 1000))
  if (seconds < 1) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

/**
 * Derive every label the connection UI needs from the store's public state.
 * Keeping this pure makes the footer and diagnostics row agree without
 * duplicating the state machine.
 */
export function liveConnectionPresentation(connectionState, reconnectAttempt, lastEventAt, now) {
  const attempt = Math.max(0, Number(reconnectAttempt) || 0)
  const ageMs = Number.isFinite(lastEventAt) ? Math.max(0, now - lastEventAt) : null
  const eventAgeLabel = ageMs == null ? 'No events yet' : `Last event ${formatAgeLabel(ageMs)}`
  const attemptLabel = `Retry attempts: ${attempt}`

  if (connectionState === 'live') {
    return {
      isLive: true,
      stateLabel: 'Live',
      statusText:
        ageMs == null ? 'Live · waiting for events' : `Live · updated ${formatAgeLabel(ageMs)}`,
      eventAgeLabel,
      attemptLabel,
      // Footer/System already render the visible state; a separate live region
      // announces only transitions so the per-second age does not spam AT.
      stateAnnouncement: '',
    }
  }

  if (connectionState === 'offline') {
    return {
      isLive: false,
      stateLabel: 'Offline',
      statusText: 'Offline — Retry',
      eventAgeLabel,
      attemptLabel,
      stateAnnouncement: 'Live updates offline. Retry available.',
    }
  }

  return {
    isLive: false,
    stateLabel: attempt === 0 ? 'Connecting' : 'Reconnecting',
    statusText: attempt === 0 ? 'Connecting…' : `Reconnecting (attempt ${attempt})…`,
    eventAgeLabel,
    attemptLabel,
    stateAnnouncement: attempt === 0 ? 'Connecting to live updates' : 'Live updates reconnecting',
  }
}

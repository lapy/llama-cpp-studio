/**
 * Shared Axios client. Cookie sessions send the CSRF header on mutations.
 * Local mode does not set the cookie, so those requests stay unchanged.
 *
 * Control-plane reads have a deadline so a stalled GET becomes a visible
 * error with Retry. Mutations are left without an automatic timeout or retry
 * because their outcome may already have been applied.
 */
import axios from 'axios'

const READ_TIMEOUT_MS = 20_000

let onSessionExpired = null

export function setSessionExpiredHandler(handler) {
  onSessionExpired = typeof handler === 'function' ? handler : null
}

export function notifySessionExpired() {
  if (onSessionExpired) onSessionExpired()
}

axios.defaults.withCredentials = true

axios.interceptors.request.use((config) => {
  const method = String(config.method || 'get').toLowerCase()
  // Axios merges its default timeout (0 = unlimited) before interceptors run.
  if ((method === 'get' || method === 'head') && (config.timeout == null || config.timeout === 0)) {
    config.timeout = READ_TIMEOUT_MS
  }
  if (method === 'get' || method === 'head' || method === 'options') return config
  if (typeof document === 'undefined') return config
  const match = document.cookie.match(/(?:^|; )studio_csrf=([^;]*)/)
  if (!match) return config
  config.headers = config.headers || {}
  config.headers['X-CSRF-Token'] = decodeURIComponent(match[1])
  return config
})

axios.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status
    const url = String(error?.config?.url || '')
    if (status === 401 && !url.includes('/api/session')) {
      notifySessionExpired()
    }
    return Promise.reject(error)
  },
)

export default axios

import { notifySessionExpired } from './client.js'

/** Same session and CSRF rules as Axios, for streamed and multipart responses. */
export async function sessionFetch(url, options = {}) {
  const headers = new Headers(options.headers)
  const method = String(options.method || 'GET').toUpperCase()
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method) && typeof document !== 'undefined') {
    const match = document.cookie.match(/(?:^|; )studio_csrf=([^;]*)/)
    if (match) headers.set('X-CSRF-Token', decodeURIComponent(match[1]))
  }
  const response = await fetch(url, { ...options, headers, credentials: 'include' })
  if (response.status === 401) notifySessionExpired()
  return response
}

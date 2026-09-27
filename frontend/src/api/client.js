/**
 * Shared Axios client. Cookie sessions send the CSRF header on mutations.
 * Local mode does not set the cookie, so those requests stay unchanged.
 */
import axios from 'axios'

axios.defaults.withCredentials = true

axios.interceptors.request.use((config) => {
  const method = String(config.method || 'get').toLowerCase()
  if (method === 'get' || method === 'head' || method === 'options') return config
  if (typeof document === 'undefined') return config
  const match = document.cookie.match(/(?:^|; )studio_csrf=([^;]*)/)
  if (!match) return config
  config.headers = config.headers || {}
  config.headers['X-CSRF-Token'] = decodeURIComponent(match[1])
  return config
})

export default axios

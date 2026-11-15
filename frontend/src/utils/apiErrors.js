/** Human-readable error detail for a failed API request. */
export function formatAxiosDetail(error) {
  const payload = error?.response?.data
  const detail = payload?.detail ?? payload
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((entry) =>
        typeof entry === 'object' && entry?.msg
          ? `${Array.isArray(entry.loc) ? entry.loc.join('.') : ''}: ${entry.msg}`.replace(
              /^\.\s*/,
              '',
            )
          : String(entry),
      )
      .filter(Boolean)
      .join('; ')
  }
  if (detail && typeof detail === 'object') {
    if (typeof detail.msg === 'string') return detail.msg
    if (typeof detail.message === 'string') return detail.message
    if (typeof detail.error === 'string') return detail.error
  }
  return error?.message || 'Request failed'
}

/** Show full release tags; shorten commit SHAs. */
export function formatEngineUpdateVersion(value) {
  const s = String(value || '').trim()
  if (/^[0-9a-f]{7,40}$/i.test(s)) return s.slice(0, 8)
  return s
}

/** Presentation of the update value advertised by an engine card. */
export const ENGINE_UPDATE_FORMATTERS = {
  plain: (value) => String(value ?? ''),
  v: (value) => `v${value ?? ''}`,
  version: (value) => formatEngineUpdateVersion(value),
}

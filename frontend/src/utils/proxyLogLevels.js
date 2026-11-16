/** Engine log filters shared by the proxy log viewer. */
export const ENGINE_LOG_LEVELS = [
  { value: '', label: 'All' },
  { value: 'warn', label: 'Warn' },
  { value: 'error', label: 'Error' },
]

/** Bounded per-model log buffer in the proxy viewer. */
export const MAX_ENGINE_LOG_LINES = 5000

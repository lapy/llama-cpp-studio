import { afterEach, beforeEach, vi } from 'vitest'

// Fake timers are process-global and can leak across files in a reused worker.
beforeEach(() => {
  vi.useRealTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

import { describe, expect, it } from 'vitest'
import { classifyPersistenceError, saveHeldForRefresh } from './persistenceOutcome'

function failed(data, status = 500) {
  return Object.assign(new Error('request failed'), { response: { status, data } })
}

describe('classifyPersistenceError', () => {
  it('keeps a full queue retryable without calling it a success', () => {
    const outcome = classifyPersistenceError(failed({
      code: 'STORE_QUEUE_FULL',
      committed: false,
    }, 503))
    expect(outcome.success).toBe(false)
    expect(outcome.preserveEdits).toBe(true)
    expect(outcome.retry).toBe(true)
    expect(outcome.refresh).toBe(false)
    expect(outcome.detail).toContain('Your edits are still here')
  })

  it('does not show success when the previous document is unchanged', () => {
    const outcome = classifyPersistenceError(failed({
      code: 'STORE_WRITE_FAILED',
      committed: false,
    }))
    expect(outcome.success).toBe(false)
    expect(outcome.committed).toBe(false)
    expect(outcome.preserveEdits).toBe(true)
    expect(outcome.retry).toBe(true)
    expect(outcome.summary).toBe('Not saved')
    expect(outcome.detail).toContain('no document was written')
  })

  it('requires a refresh after a known replacement and does not retry by itself', () => {
    const outcome = classifyPersistenceError(failed({
      code: 'STORE_WRITE_FAILED',
      committed: true,
    }))
    expect(outcome.success).toBe(false)
    expect(outcome.committed).toBe(true)
    expect(outcome.retry).toBe(false)
    expect(outcome.refresh).toBe(true)
    expect(outcome.summary).toBe('Saved, acknowledgement failed')
    expect(outcome.detail).toContain('was replaced')
  })

  it('treats a missing or non-boolean outcome as unknown', () => {
    const dropped = classifyPersistenceError(new Error('network down'))
    expect(dropped.committed).toBe('unknown')
    expect(dropped.preserveEdits).toBe(true)
    expect(dropped.retry).toBe(false)
    expect(dropped.refresh).toBe(true)

    const explicit = classifyPersistenceError(failed({
      code: 'STORE_WRITE_FAILED',
      committed: 'unknown',
    }))
    expect(explicit.committed).toBe('unknown')
    expect(explicit.success).toBe(false)
    expect(explicit.retry).toBe(false)
    expect(saveHeldForRefresh(explicit)).toBe(true)
    expect(saveHeldForRefresh({ ...explicit, refresh: false })).toBe(false)
  })
})

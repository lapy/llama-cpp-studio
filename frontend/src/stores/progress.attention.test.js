import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import axios from 'axios'
import { useProgressStore } from './progress.js'

vi.mock('axios', () => ({
  default: { get: vi.fn(), post: vi.fn() },
}))

describe('progress attention operations', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(axios.get).mockReset()
  })

  it('loads interrupted operations and clears loading', async () => {
    const rows = [{ operation_id: 'build-1', status: 'unknown' }]
    vi.mocked(axios.get).mockResolvedValue({ data: { operations: rows, count: 1 } })
    const store = useProgressStore()

    const loaded = await store.fetchAttentionOperations()

    expect(axios.get).toHaveBeenCalledWith('/api/operations/attention')
    expect(loaded).toEqual(rows)
    expect(store.attentionOperations).toEqual(rows)
    expect(store.attentionLoading).toBe(false)
    expect(store.attentionError).toBe('')
  })

  it('records a readable error and rethrows so the surface can toast it', async () => {
    vi.mocked(axios.get).mockRejectedValue({ response: { data: { detail: 'Store unreadable' } } })
    const store = useProgressStore()

    await expect(store.fetchAttentionOperations()).rejects.toBeTruthy()

    expect(store.attentionError).toBe('Store unreadable')
    expect(store.attentionLoading).toBe(false)
    expect(store.attentionOperations).toEqual([])
  })

  it('falls back to a generic error when the response has no detail', async () => {
    vi.mocked(axios.get).mockRejectedValue(new Error('network down'))
    const store = useProgressStore()

    await expect(store.fetchAttentionOperations()).rejects.toThrow('network down')

    expect(store.attentionError).toBe('network down')
  })

  it('forgets one acknowledged operation without touching the rest', () => {
    const store = useProgressStore()
    store.attentionOperations = [
      { operation_id: 'build-1', status: 'interrupted' },
      { operation_id: 'build-2', status: 'unknown' },
    ]

    store.forgetAttentionOperation('build-1')

    expect(store.attentionOperations.map((row) => row.operation_id)).toEqual(['build-2'])
    store.forgetAttentionOperation('')
    expect(store.attentionOperations).toHaveLength(1)
  })
})

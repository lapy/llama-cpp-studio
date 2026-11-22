import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

const { fetchStorageInventory, previewStorageCleanup, applyStorageCleanup } = vi.hoisted(() => ({
  fetchStorageInventory: vi.fn(),
  previewStorageCleanup: vi.fn(),
  applyStorageCleanup: vi.fn(),
}))

vi.mock('@/api/storage', () => ({
  fetchStorageInventory,
  previewStorageCleanup,
  applyStorageCleanup,
}))

import { useStorageStore } from './storage'

function inventoryFixture() {
  return {
    schema_version: 1,
    generated_at: '2026-11-22T20:00:00+00:00',
    cached: false,
    filesystem: { data_root: '/data', total_bytes: 100, used_bytes: 40, free_bytes: 60 },
    scan: { quality: 'complete', cached: false, entries_scanned: 4, notes: [] },
    categories: [
      {
        id: 'models',
        label: 'Model files',
        bytes: 30,
        unit_count: 1,
        cleanable: true,
        cleanable_bytes: 30,
        blocked: null,
        units: [],
      },
      {
        id: 'hf_cache',
        label: 'Hugging Face cache',
        bytes: 20,
        unit_count: 0,
        cleanable: true,
        cleanable_bytes: 0,
        blocked: null,
        units: [],
      },
    ],
  }
}

function planFixture() {
  return {
    plan_id: 'a'.repeat(64),
    category: 'models',
    category_label: 'Model files',
    keep_history: null,
    units: [{ id: 'u1', label: 'voice-models/orphan', bytes: 30 }],
    refused: [],
    reclaimable_bytes: 30,
    confirmation_phrase: 'DELETE 1 MODEL DIRECTORIES',
    scope_cap_units: 200,
    expires_at: '2026-11-22T20:05:00+00:00',
    notice: 'Review the selected units.',
  }
}

describe('storage store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('loads the inventory and derives categories and measured bytes', async () => {
    fetchStorageInventory.mockResolvedValueOnce(inventoryFixture())
    const store = useStorageStore()

    await store.loadInventory()

    expect(fetchStorageInventory).toHaveBeenCalledWith({ refresh: false })
    expect(store.categories).toHaveLength(2)
    expect(store.measuredBytes).toBe(50)
    expect(store.error).toBe('')
  })

  it('keeps a stable error message when the scan fails', async () => {
    fetchStorageInventory.mockRejectedValueOnce(new Error('private filesystem detail'))
    const store = useStorageStore()

    await store.loadInventory({ refresh: true })

    expect(store.inventory).toBeNull()
    expect(store.error).toBe('Storage usage could not be loaded. Retry the scan.')
  })

  it('rejects a malformed inventory payload instead of rendering a broken page', async () => {
    fetchStorageInventory.mockResolvedValueOnce('<html>not an inventory</html>')
    const store = useStorageStore()

    await store.loadInventory()

    expect(store.inventory).toBeNull()
    expect(store.error).toBe('Storage usage could not be loaded. Retry the scan.')
  })

  it('previews a cleanup and forwards the history keep count', async () => {
    previewStorageCleanup.mockResolvedValueOnce(planFixture())
    const store = useStorageStore()

    await store.previewCleanup('models', [], 25)

    expect(previewStorageCleanup).toHaveBeenCalledWith({ category: 'models', keep_history: 25 })
    expect(store.plan.plan_id).toBe('a'.repeat(64))
    expect(store.status).toBe('Review the selected units.')
  })

  it('applies a plan, reports the outcome and reloads the inventory', async () => {
    previewStorageCleanup.mockResolvedValueOnce(planFixture())
    applyStorageCleanup.mockResolvedValueOnce({
      plan_id: 'a'.repeat(64),
      category: 'models',
      category_label: 'Model files',
      outcome: 'partial',
      freed_bytes: 20,
      removed: [{ id: 'u1', label: 'voice-models/orphan', bytes: 20 }],
      failed: [
        { id: 'u2', label: 'voice-models/busy', bytes: 10, code: 'RUNNING_MODEL', message: 'x' },
      ],
      notice: 'Some units were removed.',
    })
    fetchStorageInventory.mockResolvedValue(inventoryFixture())
    const store = useStorageStore()
    await store.previewCleanup('models')

    await store.applyPlan()

    expect(applyStorageCleanup).toHaveBeenCalledWith('a'.repeat(64))
    expect(store.result.outcome).toBe('partial')
    expect(store.result.failed[0].code).toBe('RUNNING_MODEL')
    expect(store.plan).toBeNull()
    expect(store.status).toBe('Cleanup was partial. Some units were removed and the rest stayed.')
    expect(fetchStorageInventory).toHaveBeenCalledWith({ refresh: true })
  })

  it('clears a stale plan and asks for a fresh preview', async () => {
    previewStorageCleanup.mockResolvedValueOnce(planFixture())
    applyStorageCleanup.mockRejectedValueOnce({
      response: { data: { detail: { code: 'PLAN_STALE', message: 'changed' } } },
    })
    fetchStorageInventory.mockResolvedValue(inventoryFixture())
    const store = useStorageStore()
    await store.previewCleanup('models')

    await store.applyPlan()

    expect(store.plan).toBeNull()
    expect(store.error).toContain('The storage inventory changed after the preview')
    expect(fetchStorageInventory).toHaveBeenCalledWith({ refresh: true })
  })
})

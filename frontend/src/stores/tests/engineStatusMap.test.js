/**
 * R81: per-engine GET status lives in ONE id-keyed store map.
 *
 * The store used to keep six named status refs (`lmdeployStatus`,
 * `sglangStatus`, ...) selected by a `statusRef` descriptor field, with no
 * consumers outside the store. The map replaces them; `nullSafeStatus` is the
 * only remaining per-engine divergence.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useEnginesStore } from '../engines.js'
import { ENGINE_DESCRIPTORS } from '../../engines/registry.js'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

describe('keyed engine status store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('writes each engine status into the id-keyed map', async () => {
    const store = useEnginesStore()
    axios.get.mockResolvedValueOnce({ data: { active_version: 'v1' } })
    await store.venvFetchStatus('lmdeploy')
    axios.get.mockResolvedValueOnce({ data: { active_version: 'v2' } })
    await store.venvFetchStatus('1cat_vllm')
    axios.get.mockResolvedValueOnce({ data: { drifted: true } })
    await store.fetchAudioCppStatus()

    expect(store.engineStatusById.lmdeploy).toEqual({ active_version: 'v1' })
    expect(store.engineStatusById['1cat_vllm']).toEqual({ active_version: 'v2' })
    expect(store.engineStatusById.audio_cpp).toEqual({ drifted: true })
  })

  it('keeps the nullSafeStatus divergence declared per family', async () => {
    const store = useEnginesStore()
    axios.get.mockResolvedValueOnce({ data: null })
    await store.venvFetchStatus('sglang')
    expect(store.engineStatusById.sglang).toEqual({})

    axios.get.mockResolvedValueOnce({ data: null })
    await store.venvFetchStatus('lmdeploy')
    expect(store.engineStatusById.lmdeploy).toBeNull()
  })

  it('no longer declares per-engine status refs on descriptors', () => {
    for (const [engineId, descriptor] of Object.entries(ENGINE_DESCRIPTORS)) {
      expect(descriptor.statusRef, engineId).toBeUndefined()
    }
    // The keyed map is the store's exported surface for statuses.
    const store = useEnginesStore()
    expect(store.engineStatusById).toEqual({})
  })
})

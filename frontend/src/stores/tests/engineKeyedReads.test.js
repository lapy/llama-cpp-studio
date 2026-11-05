/**
 * R82: generic keyed reads for the nine local engines.
 *
 * The views keep their named refs for compatibility, but new consumers take an
 * engine id: `versionsForEngine` / `statusForEngine`. This also covers the
 * fallout from R81 (the SGLang panel read the removed named status refs).
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

describe('keyed engine reads', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('returns each engine versions list by id', () => {
    const store = useEnginesStore()
    store.sglangVersions = [{ id: 'a' }]
    store.vllmVersions = [{ id: 'b' }]
    store.audioCppVersions = [{ id: 'c' }]
    expect(store.versionsForEngine('sglang')).toEqual([{ id: 'a' }])
    expect(store.versionsForEngine('vllm')).toEqual([{ id: 'b' }])
    expect(store.versionsForEngine('audio_cpp')).toEqual([{ id: 'c' }])
    expect(store.versionsForEngine('not-an-engine')).toEqual([])
  })

  it('keeps every local engine versions list reachable through its descriptor key', () => {
    // R84/R89: no engine-id -> ref table in the store; one keyed map with
    // writable aliases. This walks the real descriptors.
    const store = useEnginesStore()
    const engines = Object.entries(ENGINE_DESCRIPTORS)
    expect(engines.length).toBeGreaterThan(0)
    engines.forEach(([engineId, descriptor], index) => {
      expect(descriptor.versionsKey, engineId).toBeTruthy()
      expect(store[descriptor.versionsKey], engineId).toBeDefined()
      store[descriptor.versionsKey] = [{ id: `row-${index}` }]
      expect(store.versionsForEngine(engineId), engineId).toEqual([{ id: `row-${index}` }])
      // The alias and the keyed map are the same data in both directions.
      expect(store.engineVersionsById[engineId], engineId).toEqual([{ id: `row-${index}` }])
      store.engineVersionsById[engineId] = [{ id: `map-${index}` }]
      expect(store[descriptor.versionsKey], engineId).toEqual([{ id: `map-${index}` }])
    })
  })

  it('partitions fetched rows into the keyed map', async () => {
    const store = useEnginesStore()
    axios.get.mockResolvedValueOnce({
      data: [
        { id: 'a', repository_source: 'llama.cpp', version: '1' },
        { id: 'b', repository_source: 'LMDeploy', version: '2' },
      ],
    })
    await store.fetchLlamaVersions()
    expect(store.engineVersionsById.llama_cpp.map((v) => v.id)).toEqual(['a'])
    expect(store.engineVersionsById.lmdeploy.map((v) => v.id)).toEqual(['b'])
    expect(store.llamaVersions.map((v) => v.id)).toEqual(['a'])
    expect(store.lmdeployVersions.map((v) => v.id)).toEqual(['b'])
  })

  it('returns the keyed status by id with an empty fallback', () => {
    const store = useEnginesStore()
    expect(store.statusForEngine('sglang')).toEqual({})
    axios.get.mockResolvedValueOnce({ data: { installed: true } })
    return store.venvFetchStatus('sglang').then(() => {
      expect(store.statusForEngine('sglang')).toEqual({ installed: true })
      expect(store.statusForEngine('vllm')).toEqual({})
    })
  })
})

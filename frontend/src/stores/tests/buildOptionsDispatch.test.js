/**
 * R95: the build-options catalog fetch dispatches on the engine's DECLARED
 * `buildOptionsPath` instead of engine-id branches in the views.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useEnginesStore } from '../engines.js'
import { buildOptionsPath } from '../../engines/registry.js'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

describe('build-options dispatch', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('declares a catalog path for the engines that have one', () => {
    expect(buildOptionsPath('llama_cpp')).toBe('/api/llama-versions/build-options')
    expect(buildOptionsPath('ik_llama')).toBe('/api/llama-versions/build-options')
    expect(buildOptionsPath('audio_cpp')).toBe('/api/audio-cpp/build-options')
    expect(buildOptionsPath('unsloth_llama')).toBeNull()
    expect(buildOptionsPath('not-an-engine')).toBeNull()
  })

  it('fetches from the declared path', async () => {
    const store = useEnginesStore()
    axios.get.mockResolvedValueOnce({ data: { categories: [] } })
    await store.fetchBuildOptionsForEngine('audio_cpp')
    expect(axios.get).toHaveBeenCalledWith('/api/audio-cpp/build-options', {
      params: { engine: 'audio_cpp' },
    })

    axios.get.mockResolvedValueOnce({ data: { categories: [] } })
    await store.fetchBuildOptionsForEngine('ik_llama')
    expect(axios.get).toHaveBeenCalledWith('/api/llama-versions/build-options', {
      params: { engine: 'ik_llama' },
    })
  })

  it('keeps the compat wrapper on the declared path too', async () => {
    const store = useEnginesStore()
    axios.get.mockResolvedValueOnce({ data: { categories: ['audio'] } })
    const result = await store.fetchAudioCppBuildOptions()
    expect(axios.get).toHaveBeenCalledWith('/api/audio-cpp/build-options', {
      params: { engine: 'audio_cpp' },
    })
    expect(result).toEqual({ categories: ['audio'] })
  })
})

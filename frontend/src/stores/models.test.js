import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'

const markSwapConfigStaleLocal = vi.fn()
const fetchSwapConfigStale = vi.fn()

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    markSwapConfigStaleLocal,
    fetchSwapConfigStale,
  }),
}))

import { useModelStore } from './models.js'

describe('models store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    markSwapConfigStaleLocal.mockReset()
    fetchSwapConfigStale.mockReset()
    vi.mocked(axios.get).mockReset()
    vi.mocked(axios.post).mockReset()
    vi.mocked(axios.put).mockReset()
    vi.mocked(axios.delete).mockReset()
  })

  it('fetchModels populates grouped models and flattened quantizations', async () => {
    vi.mocked(axios.get).mockResolvedValue({
      data: [
        {
          huggingface_id: 'org/model',
          base_model_name: 'Model',
          model_type: 'llama',
          pipeline_tag: 'text-generation',
          is_embedding_model: false,
          quantizations: [
            {
              id: 'm1',
              quantization: 'Q4_K_M',
              downloaded_at: '2026-01-01T00:00:00Z',
              is_active: true,
            },
          ],
        },
      ],
    })

    const store = useModelStore()
    await store.fetchModels()

    expect(store.models).toHaveLength(1)
    expect(store.allQuantizations).toHaveLength(1)
    expect(store.allQuantizations[0]).toMatchObject({
      id: 'm1',
      base_model_name: 'Model',
      huggingface_id: 'org/model',
      model_type: 'llama',
      pipeline_tag: 'text-generation',
    })
    expect(store.downloadedModels).toHaveLength(1)
    expect(store.runningModels).toHaveLength(1)
  })

  it('updateModelConfig notifies stale state immediately', async () => {
    vi.mocked(axios.put).mockResolvedValue({ data: {} })
    fetchSwapConfigStale.mockResolvedValue({ applicable: true, stale: true })

    const store = useModelStore()
    await store.updateModelConfig('org/model', { engine: 'llama_cpp' })

    expect(axios.put).toHaveBeenCalledWith('/api/models/org%2Fmodel/config', {
      engine: 'llama_cpp',
    })
    expect(markSwapConfigStaleLocal).toHaveBeenCalledTimes(1)
    expect(fetchSwapConfigStale).toHaveBeenCalledTimes(1)
  })

  it('deleteModel refetches models and marks stale', async () => {
    vi.mocked(axios.delete).mockResolvedValue({ data: {} })
    vi.mocked(axios.get).mockResolvedValue({ data: [] })
    fetchSwapConfigStale.mockResolvedValue({ applicable: true, stale: true })

    const store = useModelStore()
    await store.deleteModel('org/model')

    expect(axios.delete).toHaveBeenCalledWith('/api/models/org%2Fmodel')
    expect(axios.get).toHaveBeenCalledWith('/api/models')
    expect(markSwapConfigStaleLocal).toHaveBeenCalledTimes(1)
  })

  it('searchModels clears results and loading on failure', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.mocked(axios.post).mockRejectedValue(new Error('search down'))

    const store = useModelStore()
    await expect(store.searchModels('qwen')).rejects.toThrow('search down')

    expect(store.searchLoading).toBe(false)
    expect(store.searchResults).toEqual([])
  })

  it('fetches safetensors models and quantization sizes from live routes', async () => {
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/models/safetensors') {
        return Promise.resolve({ data: [{ huggingface_id: 'org/repo' }] })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    vi.mocked(axios.post).mockImplementation((url) => {
      if (url === '/api/models/quantization-sizes') {
        return Promise.resolve({
          data: {
            quantizations: {
              Q4_K_M: { filename: 'model-Q4_K_M.gguf', size: 123 },
            },
          },
        })
      }
      throw new Error(`Unexpected POST ${url}`)
    })

    const store = useModelStore()
    await store.fetchSafetensorsModels()
    const sizes = await store.getQuantizationSizes('org/repo', {
      Q4_K_M: { filename: 'model-Q4_K_M.gguf' },
    })

    expect(store.safetensorsModels).toEqual([{ huggingface_id: 'org/repo' }])
    expect(sizes).toEqual({
      Q4_K_M: { filename: 'model-Q4_K_M.gguf', size: 123 },
    })
  })

  it('supports group deletion, projector updates, token refresh, and search reset', async () => {
    vi.mocked(axios.post).mockImplementation((url, body) => {
      if (url === '/api/models/delete-group') {
        return Promise.resolve({ data: { ok: true } })
      }
      if (url === '/api/models/org%2Frepo/projector') {
        return Promise.resolve({ data: { mmproj_filename: body.mmproj_filename } })
      }
      if (url === '/api/models/huggingface-token') {
        return Promise.resolve({ data: { ok: true } })
      }
      throw new Error(`Unexpected POST ${url}`)
    })
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/models') {
        return Promise.resolve({ data: [] })
      }
      if (url === '/api/models/huggingface-token') {
        return Promise.resolve({
          data: {
            has_token: true,
            token_preview: 'hf_xxxx...',
            from_environment: false,
          },
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useModelStore()
    store.searchQuery = 'before'
    store.searchLastQuery = 'before'
    store.searchHasSearched = true
    store.searchResults = [{ id: 1 }]

    await store.deleteModelGroup('org/repo')
    const projector = await store.updateModelProjector('org/repo', 'mmproj.gguf', 123)
    const tokenSet = await store.setHuggingfaceToken('hf_secret_value')
    const tokenCleared = await store.clearHuggingfaceToken()
    store.clearSearchState()

    expect(projector).toEqual({ mmproj_filename: 'mmproj.gguf' })
    expect(tokenSet).toEqual({ ok: true })
    expect(tokenCleared).toEqual({ ok: true })
    expect(store.hasHuggingfaceToken).toBe(true)
    expect(store.huggingfaceToken).toBe('hf_xxxx...')
    expect(store.searchQuery).toBe('')
    expect(store.searchLastQuery).toBe('')
    expect(store.searchHasSearched).toBe(false)
    expect(store.searchResults).toEqual([])
    expect(markSwapConfigStaleLocal).toHaveBeenCalled()
  })
})

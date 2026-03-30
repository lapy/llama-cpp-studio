import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useEnginesStore } from './engines.js'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  },
}))

describe('engines store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(axios.get).mockReset()
    vi.mocked(axios.post).mockReset()
    vi.mocked(axios.put).mockReset()
  })

  it('fetchLlamaVersions partitions rows by repository_source', async () => {
    vi.mocked(axios.get).mockResolvedValue({
      data: [
        { id: 'a', repository_source: 'llama.cpp', version: '1' },
        { id: 'b', repository_source: 'ik_llama.cpp', version: '2' },
        { id: 'c', repository_source: 'LMDeploy', version: '3' },
      ],
    })

    const store = useEnginesStore()
    await store.fetchLlamaVersions()

    expect(store.llamaVersions).toHaveLength(1)
    expect(store.llamaVersions[0].id).toBe('a')
    expect(store.ikLlamaVersions).toHaveLength(1)
    expect(store.lmdeployVersions).toHaveLength(1)
    expect(axios.get).toHaveBeenCalledWith('/api/llama-versions')
  })

  it('cancelSourceBuild posts task id', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { ok: true } })
    const store = useEnginesStore()
    await store.cancelSourceBuild('task-xyz')
    expect(axios.post).toHaveBeenCalledWith('/api/llama-versions/build-cancel', {
      task_id: 'task-xyz',
    })
  })

  it('applySwapConfig clears stale immediately and refreshes in background', async () => {
    let resolveStale
    let resolveStatus
    let resolveGpu

    vi.mocked(axios.post).mockResolvedValue({ data: { ok: true } })
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/llama-swap/stale') {
        return new Promise((resolve) => {
          resolveStale = resolve
        })
      }
      if (url === '/api/status') {
        return new Promise((resolve) => {
          resolveStatus = resolve
        })
      }
      if (url === '/api/gpu-info') {
        return new Promise((resolve) => {
          resolveGpu = resolve
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    store.swapConfigStale = { applicable: true, stale: true }

    await store.applySwapConfig()

    expect(store.swapConfigStale).toEqual({ applicable: true, stale: false })
    expect(axios.post).toHaveBeenCalledWith('/api/llama-swap/apply-config')
    expect(axios.get).toHaveBeenCalledWith('/api/llama-swap/stale')
    expect(axios.get).toHaveBeenCalledWith('/api/status')
    expect(axios.get).toHaveBeenCalledWith('/api/gpu-info')

    resolveStale({ data: { applicable: true, stale: false } })
    resolveStatus({ data: {} })
    resolveGpu({ data: {} })
  })

  it('fetchSwapConfigStale deduplicates concurrent requests', async () => {
    let resolveStale
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/llama-swap/stale') {
        return new Promise((resolve) => {
          resolveStale = resolve
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    const first = store.fetchSwapConfigStale()
    const second = store.fetchSwapConfigStale()

    expect(axios.get).toHaveBeenCalledTimes(1)

    resolveStale({ data: { applicable: true, stale: true } })
    const [firstResult, secondResult] = await Promise.all([first, second])

    expect(firstResult).toEqual({ applicable: true, stale: true })
    expect(secondResult).toEqual({ applicable: true, stale: true })
    expect(store.swapConfigStale).toEqual({ applicable: true, stale: true })
  })

  it('local stale mark wins over an older in-flight stale fetch', async () => {
    let resolveStale
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/llama-swap/stale') {
        return new Promise((resolve) => {
          resolveStale = resolve
        })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    const pending = store.fetchSwapConfigStale()
    store.markSwapConfigStaleLocal()

    expect(store.swapConfigStale).toEqual({ applicable: true, stale: true })

    resolveStale({ data: { applicable: true, stale: false } })
    await pending

    expect(store.swapConfigStale).toEqual({ applicable: true, stale: true })
  })

  it('activateVersion refreshes lmdeploy state for lmdeploy versions', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { ok: true } })
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/llama-versions') {
        return Promise.resolve({ data: [] })
      }
      if (url === '/api/lmdeploy/status') {
        return Promise.resolve({ data: { installed: true } })
      }
      if (url === '/api/llama-swap/stale') {
        return Promise.resolve({ data: { applicable: true, stale: false } })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    await store.activateVersion('lmdeploy-v1')

    expect(axios.post).toHaveBeenCalledWith('/api/llama-versions/versions/activate', {
      version_id: 'lmdeploy-v1',
    })
    expect(axios.get).toHaveBeenCalledWith('/api/lmdeploy/status')
    expect(store.lmdeployStatus).toEqual({ installed: true })
  })

  it('fetchSystemStatus rethrows failures and clears loading', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/status') {
        return Promise.reject(new Error('status down'))
      }
      if (url === '/api/gpu-info') {
        return Promise.resolve({ data: {} })
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    await expect(store.fetchSystemStatus()).rejects.toThrow('status down')
    expect(store.loading).toBe(false)
  })

  it('fetchSwapConfigPending keeps the previous default on failure', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.mocked(axios.get).mockRejectedValue(new Error('pending failed'))

    const store = useEnginesStore()
    const result = await store.fetchSwapConfigPending()

    expect(result).toEqual({
      applicable: false,
      pending: false,
      changes: [],
      reason: null,
    })
  })

  it('fetchAll settles every request without throwing', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/llama-versions') {
        return Promise.resolve({ data: [] })
      }
      if (url === '/api/llama-versions/cuda-status') {
        return Promise.reject(new Error('cuda down'))
      }
      if (url === '/api/lmdeploy/status') {
        return Promise.resolve({ data: { installed: false } })
      }
      if (url === '/api/status') {
        return Promise.resolve({ data: { proxy_status: { healthy: true } } })
      }
      if (url === '/api/gpu-info') {
        return Promise.resolve({ data: { cpu_threads: 8 } })
      }
      if (url === '/api/llama-swap/stale') {
        return Promise.reject(new Error('stale failed'))
      }
      throw new Error(`Unexpected GET ${url}`)
    })

    const store = useEnginesStore()
    await expect(store.fetchAll()).resolves.toBeUndefined()
    expect(store.systemStatus).toEqual({ proxy_status: { healthy: true } })
    expect(store.gpuInfo).toEqual({ cpu_threads: 8 })
  })
})

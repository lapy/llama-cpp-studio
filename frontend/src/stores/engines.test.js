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
})

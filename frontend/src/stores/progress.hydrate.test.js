import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import axios from 'axios'
import { useProgressStore } from './progress.js'

vi.mock('axios', () => ({
  default: { get: vi.fn() },
}))

describe('progress log restoration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(axios.get).mockReset()
  })

  it('replaces the reconnect tail with the full retained server log', async () => {
    const store = useProgressStore()
    store.taskLogs = { install_1: ['last 100 lines only'] }
    vi.mocked(axios.get).mockResolvedValue({
      data: { lines: ['first line', 'middle line', 'last 100 lines only'] },
    })

    const lines = await store.hydrateTaskLogs('install_1')

    expect(axios.get).toHaveBeenCalledWith('/api/operations/tasks/install_1/logs')
    expect(lines).toEqual(['first line', 'middle line', 'last 100 lines only'])
    expect(store.getTaskLogs('install_1')).toEqual(lines)
  })

  it('keeps live output when the task is no longer retained', async () => {
    const store = useProgressStore()
    store.taskLogs = { install_2: ['streamed while open'] }
    vi.mocked(axios.get).mockRejectedValue(new Error('not found'))

    const lines = await store.hydrateTaskLogs('install_2')

    expect(lines).toEqual(['streamed while open'])
    expect(store.getTaskLogs('install_2')).toEqual(['streamed while open'])
  })

  it('hydrates each task only once', async () => {
    const store = useProgressStore()
    vi.mocked(axios.get).mockResolvedValue({ data: { lines: ['one'] } })

    await store.hydrateTaskLogs('install_3')
    await store.hydrateTaskLogs('install_3')

    expect(axios.get).toHaveBeenCalledTimes(1)
  })
})

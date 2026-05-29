import { describe, it, expect, beforeEach, vi } from 'vitest'
import { ref } from 'vue'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import { useTaskActions } from './useTaskActions.js'
import { useProgressStore } from '@/stores/progress'
import { REAL_TASK_FIXTURES } from '@/test-fixtures/taskFixtures.js'

const toastAdd = vi.fn()

vi.mock('axios', () => ({
  default: { post: vi.fn() },
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

describe('useTaskActions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    toastAdd.mockReset()
    axios.post.mockReset()
    axios.post.mockResolvedValue({ data: { ok: true, message: 'Stopped' } })
  })

  it.each(REAL_TASK_FIXTURES)('canStopTask is true for running $label', ({ task }) => {
    const { canStopTask } = useTaskActions()
    expect(canStopTask(task)).toBe(true)
  })

  it.each(REAL_TASK_FIXTURES)(
    'requestStopTask posts to the correct cancel endpoint for $label',
    async ({ task, cancelEndpoint }) => {
      const { requestStopTask } = useTaskActions()
      await requestStopTask(task)
      expect(axios.post).toHaveBeenCalledWith(cancelEndpoint, { task_id: task.task_id })
      expect(toastAdd).toHaveBeenCalledWith(
        expect.objectContaining({ severity: 'info', summary: 'Stop requested' }),
      )
    },
  )

  it('canStopTask is false for completed tasks even when cancel endpoint exists', () => {
    const { canStopTask } = useTaskActions()
    const task = { ...REAL_TASK_FIXTURES[0].task, status: 'completed' }
    expect(canStopTask(task)).toBe(false)
  })

  it('canStopTask is false when cancel endpoint is unavailable', () => {
    const { canStopTask } = useTaskActions()
    expect(canStopTask({ task_id: 'x', type: 'unknown', status: 'running' })).toBe(false)
  })

  it('requestStopTask shows warning when API returns ok false', async () => {
    axios.post.mockResolvedValue({ data: { ok: false, message: 'Already finished' } })
    const { requestStopTask } = useTaskActions()
    await requestStopTask(REAL_TASK_FIXTURES[0].task)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'warn', detail: 'Already finished' }),
    )
  })

  it('requestStopTask shows error toast on HTTP failure', async () => {
    axios.post.mockRejectedValue({
      response: { data: { detail: 'Server error' } },
      message: 'Network',
    })
    const { requestStopTask } = useTaskActions()
    await requestStopTask(REAL_TASK_FIXTURES[0].task)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'error', detail: 'Server error' }),
    )
  })

  it('getTaskLogs returns logs from the progress store', () => {
    const store = useProgressStore()
    store.taskLogs = { build_1: ['line a', 'line b'] }
    const { getTaskLogs } = useTaskActions()
    expect(getTaskLogs({ task_id: 'build_1' })).toEqual(['line a', 'line b'])
  })

  it('dismissTask removes task, expanded state, and log element refs', () => {
    const store = useProgressStore()
    store.tasks = { t1: { task_id: 't1', status: 'completed' } }
    store.taskLogs = { t1: ['log'] }
    const expandedLogsRef = ref({ t1: true, t2: true })
    const logPreEls = { t1: {}, t2: {} }

    const { dismissTask } = useTaskActions()
    dismissTask('t1', expandedLogsRef, logPreEls)

    expect(store.getTask('t1')).toBeNull()
    expect(expandedLogsRef.value).toEqual({ t2: true })
    expect(logPreEls.t1).toBeUndefined()
    expect(logPreEls.t2).toBeDefined()
  })
})

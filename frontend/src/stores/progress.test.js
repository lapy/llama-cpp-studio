import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useProgressStore } from './progress.js'

describe('progress store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('getTask returns null for unknown id', () => {
    const store = useProgressStore()
    expect(store.getTask('missing')).toBeNull()
  })

  it('removeTask drops task and logs', () => {
    const store = useProgressStore()
    Object.assign(store.tasks, {
      a: {
        task_id: 'a',
        status: 'completed',
        description: 'done',
      },
    })
    Object.assign(store.taskLogs, { a: ['line1'] })
    store.removeTask('a')
    expect(store.getTask('a')).toBeNull()
    expect(store.getTaskLogs('a')).toEqual([])
  })

  it('subscribe returns unsubscribe function', () => {
    const store = useProgressStore()
    const off = store.subscribe('notification', () => {})
    expect(typeof off).toBe('function')
    off()
  })
})

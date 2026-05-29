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
      a: { task_id: 'a', status: 'completed', description: 'done' },
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

  it('handleEvent stores task_created by real task_id', () => {
    const store = useProgressStore()
    const task = {
      task_id: 'download_gguf_org_model_Q4_K_M_1',
      type: 'download',
      status: 'running',
      progress: 0,
      description: 'Download',
      metadata: { huggingface_id: 'org/model' },
    }

    store.handleEvent('task_created', task)

    expect(store.getTask('download_gguf_org_model_Q4_K_M_1')).toEqual(task)
    expect(Object.keys(store.tasks)).toEqual(['download_gguf_org_model_Q4_K_M_1'])
  })

  it('handleEvent updates task_updated without changing task_id key', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'build_sync_main_1',
      type: 'build',
      status: 'running',
      progress: 10,
      description: 'Sync',
    })
    store.handleEvent('task_updated', {
      task_id: 'build_sync_main_1',
      type: 'build',
      status: 'completed',
      progress: 100,
      description: 'Sync done',
    })

    expect(store.getTask('build_sync_main_1')?.status).toBe('completed')
  })

  it('handleEvent appends task_log lines to the matching task_id', () => {
    const store = useProgressStore()
    store.handleEvent('task_log', { task_id: 'install_lmdeploy_install_1', line: 'pip install' })
    store.handleEvent('task_log', { task_id: 'install_lmdeploy_install_1', line: 'done' })
    expect(store.getTaskLogs('install_lmdeploy_install_1')).toEqual(['pip install', 'done'])
  })

  it('handleEvent appends legacy install logs by task_id', () => {
    const store = useProgressStore()
    store.handleEvent('lmdeploy_install_log', {
      task_id: 'install_lmdeploy_install_1',
      line: 'legacy line',
    })
    expect(store.getTaskLogs('install_lmdeploy_install_1')).toEqual(['legacy line'])
  })

  it('handleEvent appends build_progress log lines without dedupe', () => {
    const store = useProgressStore()
    store.handleEvent('build_progress', {
      task_id: 'build_source-main_1',
      log_lines: ['cc -o main', 'cc -o main'],
    })
    expect(store.getTaskLogs('build_source-main_1')).toEqual(['cc -o main', 'cc -o main'])
  })

  it('handleEvent syncs metadata log_lines on task_created', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'build_1',
      type: 'build',
      status: 'running',
      progress: 0,
      description: 'Build',
      metadata: { log_lines: ['cmake ..'] },
    })
    expect(store.getTaskLogs('build_1')).toEqual(['cmake ..'])
  })

  it('handleEvent notifies subscribers for task_updated', () => {
    const store = useProgressStore()
    const seen = []
    store.subscribe('task_updated', (payload) => seen.push(payload))

    const task = {
      task_id: 'install_cuda_install_1',
      type: 'install',
      status: 'running',
      metadata: { manager: 'cuda' },
    }
    store.handleEvent('task_updated', task)

    expect(seen).toHaveLength(1)
    expect(seen[0].task_id).toBe('install_cuda_install_1')
  })

  it('handleEvent ignores invalid JSON payloads', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', '{not json')
    expect(Object.keys(store.tasks)).toHaveLength(0)
  })

  it('activeTasks includes only running tasks', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'run',
      type: 'download',
      status: 'running',
      progress: 1,
    })
    store.handleEvent('task_created', {
      task_id: 'done',
      type: 'download',
      status: 'completed',
      progress: 100,
    })
    expect(store.activeTasks.map((t) => t.task_id)).toEqual(['run'])
  })
})

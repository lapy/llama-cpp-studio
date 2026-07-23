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

  it('handleEvent dedupes mirrored build_progress log lines', () => {
    const store = useProgressStore()
    store.handleEvent('build_progress', {
      task_id: 'build_source-main_1',
      log_lines: ['cc -o main', 'cc -o main'],
    })
    expect(store.getTaskLogs('build_source-main_1')).toEqual(['cc -o main'])
  })

  it('handleEvent does not double-append build logs from task_updated and build_progress', () => {
    const store = useProgressStore()
    store.handleEvent('task_updated', {
      task_id: 'build_source-main_1',
      type: 'build',
      status: 'running',
      progress: 40,
      metadata: { log_lines: ['Starting compilation...', '[ 46%] Built target ggml-cuda'] },
    })
    store.handleEvent('build_progress', {
      task_id: 'build_source-main_1',
      stage: 'build',
      progress: 40,
      log_lines: ['Starting compilation...', '[ 46%] Built target ggml-cuda'],
    })
    expect(store.getTaskLogs('build_source-main_1')).toEqual([
      'Starting compilation...',
      '[ 46%] Built target ggml-cuda',
    ])

    store.handleEvent('task_updated', {
      task_id: 'build_source-main_1',
      type: 'build',
      status: 'running',
      progress: 90,
      metadata: { log_lines: ['Validation: Passed'] },
    })
    store.handleEvent('build_progress', {
      task_id: 'build_source-main_1',
      stage: 'validate',
      progress: 90,
      log_lines: ['Validation: Passed'],
    })
    expect(store.getTaskLogs('build_source-main_1')).toEqual([
      'Starting compilation...',
      '[ 46%] Built target ggml-cuda',
      'Validation: Passed',
    ])
  })

  it('handleEvent applies build_progress percent to the matching task', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'build_source-main_1',
      type: 'build',
      status: 'running',
      progress: 72,
      description: 'Build llama.cpp',
    })
    store.handleEvent('build_progress', {
      task_id: 'build_source-main_1',
      stage: 'build',
      progress: 84,
      message: 'Building llama.cpp [120/200]',
      log_lines: ['[120/200] Building CXX object foo.cpp.o'],
    })
    const task = store.getTask('build_source-main_1')
    expect(task?.progress).toBe(84)
    expect(task?.message).toBe('Building llama.cpp [120/200]')
    expect(task?.metadata?.stage).toBe('build')
    expect(store.getTaskLogs('build_source-main_1')).toEqual([
      '[120/200] Building CXX object foo.cpp.o',
    ])
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

  it('handleEvent merges new metadata log_lines on task_updated', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'audio_install_1',
      type: 'audio_model_install',
      status: 'running',
      progress: 10,
      description: 'Install',
      metadata: { log_lines: ['Downloading package'] },
    })
    store.handleEvent('task_updated', {
      task_id: 'audio_install_1',
      type: 'audio_model_install',
      status: 'running',
      progress: 40,
      message: 'Extracting',
      metadata: { log_lines: ['Downloading package', 'Extracting'] },
    })
    expect(store.getTaskLogs('audio_install_1')).toEqual([
      'Downloading package',
      'Extracting',
    ])
  })

  it('handleEvent applies download_progress to the matching task', () => {
    const store = useProgressStore()
    store.handleEvent('task_created', {
      task_id: 'download_1',
      type: 'download',
      status: 'running',
      progress: 0,
      description: 'Download model',
    })
    store.handleEvent('download_progress', {
      task_id: 'download_1',
      progress: 42,
      message: 'Downloading model.gguf (420.0/1000.0 MB, 12.5 MB/s)',
      bytes_downloaded: 420 * 1024 * 1024,
      total_bytes: 1000 * 1024 * 1024,
      speed_mbps: 12.5,
      filename: 'model.gguf',
    })

    const task = store.getTask('download_1')
    expect(task?.progress).toBe(42)
    expect(task?.message).toContain('12.5 MB/s')
    expect(task?.metadata?.bytes_downloaded).toBe(420 * 1024 * 1024)
    expect(task?.metadata?.total_bytes).toBe(1000 * 1024 * 1024)
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

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { reactive } from 'vue'
import axios from 'axios'
import TaskNotifications from './TaskNotifications.vue'
import { useProgressStore } from '@/stores/progress'
import { REAL_TASK_FIXTURES } from '@/test-fixtures/taskFixtures.js'

vi.mock('axios', () => ({
  default: {
    post: vi.fn(),
  },
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: vi.fn() }),
}))

describe('TaskNotifications', () => {
  beforeEach(() => {
    localStorage.removeItem('llama-studio.activity.dismissed')
    setActivePinia(createPinia())
    axios.post.mockReset()
    axios.post.mockResolvedValue({ data: { ok: true } })
  })

  function mountTray() {
    return mount(TaskNotifications, {
      global: {
        stubs: {
          Teleport: true,
          Dialog: {
            props: ['visible'],
            template: '<div class="dialog-stub" v-if="visible"><slot /></div>',
          },
          Button: { template: '<button><slot /></button>' },
          ProgressBar: {
            props: ['value'],
            template: '<div class="progress-bar-stub">{{ value }}</div>',
          },
          TaskDetailPanel: {
            props: ['taskId'],
            template: '<div class="detail-panel-stub">{{ taskId }}</div>',
          },
        },
      },
      attachTo: document.body,
    })
  }

  function seedTask(task) {
    const store = useProgressStore()
    store.tasks = { [task.task_id]: task }
    return store
  }

  it('renders toast cards for active tasks', async () => {
    seedTask({
      task_id: 'dl',
      type: 'download',
      status: 'running',
      progress: 42,
      description: 'Downloading model.gguf',
      message: 'Downloading model.gguf (420.0/1000.0 MB)',
      metadata: {
        bytes_downloaded: 420_000_000,
        total_bytes: 1_000_000_000,
        files_completed: 2,
        files_total: 4,
      },
    })

    const wrapper = mountTray()
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading model.gguf')
    expect(wrapper.text()).toContain('42%')
    expect(wrapper.find('.task-toast__message').text().trim()).toBe('')
    expect(wrapper.find('.task-toast__download-meta').text()).toBe('420 MB / 1.0 GB · 580 MB left · file 2/4')
  })

  it('shows each task once inside the activity panel', async () => {
    seedTask({
      task_id: 'build',
      type: 'build',
      status: 'running',
      progress: 10,
      description: 'Building llama.cpp',
    })

    const wrapper = mountTray()
    await flushPromises()

    expect(wrapper.find('.task-notifications-tray').exists()).toBe(false)
    expect(wrapper.find('.detail-panel-stub').exists()).toBe(false)
    expect(wrapper.findAll('.activity-panel .task-toast')).toHaveLength(1)
    expect(wrapper.find('.activity-panel').text()).toContain('Building llama.cpp')
    expect(wrapper.find('.activity-panel--alert').exists()).toBe(true)
  })

  it('reopens the activity panel when another task starts or one fails', async () => {
    const store = useProgressStore()
    const wrapper = mountTray()
    await flushPromises()

    expect(wrapper.find('.activity-panel').exists()).toBe(false)

    store.tasks = reactive({
      build: {
        task_id: 'build',
        type: 'build',
        status: 'running',
        progress: 4,
        description: 'Building llama.cpp',
      },
    })
    await flushPromises()

    expect(wrapper.find('.activity-panel').text()).toContain('Building llama.cpp')
    expect(wrapper.get('.activity-toggle').attributes('aria-expanded')).toBe('true')

    await wrapper.get('.activity-toggle').trigger('click')
    expect(wrapper.find('.activity-panel').exists()).toBe(false)

    store.tasks = reactive({
      build: store.tasks.build,
      install: {
        task_id: 'install',
        type: 'install',
        status: 'queued',
        progress: 0,
        description: 'Installing engine',
      },
    })
    await flushPromises()

    expect(wrapper.find('.activity-panel').text()).toContain('Installing engine')

    await wrapper.get('.activity-toggle').trigger('click')
    expect(wrapper.find('.activity-panel').exists()).toBe(false)

    store.tasks.build.status = 'failed'
    store.tasks.build.error = 'protoc failed'
    await flushPromises()

    expect(wrapper.find('.activity-panel').text()).toContain('protoc failed')
    wrapper.unmount()
  })

  it('dismisses finished tasks from the tray', async () => {
    const store = useProgressStore()
    store.tasks = reactive({
      done: {
        task_id: 'done',
        type: 'download',
        status: 'completed',
        progress: 100,
        description: 'Download complete',
      },
    })

    const wrapper = mountTray()
    await flushPromises()
    await wrapper.get('.activity-toggle').trigger('click')
    await flushPromises()

    await wrapper.find('.task-toast__dismiss').trigger('click')
    await flushPromises()

    expect(store.getTask('done')).toBeNull()
    expect(wrapper.find('.task-toast').exists()).toBe(false)
    expect(axios.post).toHaveBeenCalledWith('/api/tasks/dismiss', { task_id: 'done' })

    store.handleEvent('task_snapshot', {
      tasks: [{
        task_id: 'done',
        type: 'download',
        status: 'completed',
        progress: 100,
        description: 'Download complete',
      }],
    })
    await flushPromises()
    expect(wrapper.find('.task-toast').exists()).toBe(false)
  })

  it('leaves housekeeping and recovered successes out of activity', async () => {
    const store = useProgressStore()
    store.tasks = {
      scan: {
        task_id: 'scan',
        type: 'param_scan',
        status: 'completed',
        progress: 100,
        description: 'Scan llama.cpp CLI parameters',
        message: 'Indexed 250 CLI options',
      },
      apply: {
        task_id: 'apply',
        type: 'runtime_apply',
        status: 'completed',
        progress: 100,
        description: 'runtime_apply',
        metadata: { recovered: true },
      },
      oldSync: {
        task_id: 'old-sync',
        type: 'build',
        status: 'completed',
        progress: 100,
        description: 'build',
        message: 'Synced source',
        metadata: { recovered: true },
      },
      sync: {
        task_id: 'sync',
        type: 'build',
        status: 'completed',
        progress: 100,
        description: 'Sync llama.cpp master',
      },
    }

    const wrapper = mountTray()
    await flushPromises()
    await wrapper.get('.activity-toggle').trigger('click')
    await flushPromises()

    const titles = wrapper.findAll('.task-toast__title').map((node) => node.text())
    expect(titles).toEqual(['Sync llama.cpp master'])
  })

  it.each(REAL_TASK_FIXTURES)(
    'requests cancellation via $cancelEndpoint for $label',
    async ({ task, cancelEndpoint }) => {
      seedTask(task)
      const wrapper = mountTray()
      await flushPromises()

      await wrapper.find('.task-toast__stop').trigger('click')
      await flushPromises()

      expect(axios.post).toHaveBeenCalledWith(cancelEndpoint, {
        task_id: task.task_id,
      })
    },
  )

  it('does not show stop button for completed tasks', async () => {
    seedTask({
      ...REAL_TASK_FIXTURES[0].task,
      status: 'completed',
      progress: 100,
    })

    const wrapper = mountTray()
    await flushPromises()
    await wrapper.get('.activity-toggle').trigger('click')
    await flushPromises()

    expect(wrapper.find('.task-toast__stop').exists()).toBe(false)
  })

  it('opens a log viewer that copies text and follows new lines until scrolled away', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const store = seedTask({
      task_id: 'build',
      type: 'build',
      status: 'running',
      progress: 20,
      description: 'Building llama.cpp',
    })
    store.taskLogs = { build: ['cmake ..', 'make -j'] }

    const wrapper = mountTray()
    await flushPromises()
    await wrapper.get('.task-toast__body').trigger('click')
    await flushPromises()

    const log = wrapper.get('.task-toast__logs')
    expect(log.text()).toContain('cmake ..')
    expect(log.text()).toContain('make -j')
    expect(wrapper.get('.task-toast__log-follow').text()).toBe('Following')

    await wrapper.get('.task-toast__log-copy').trigger('click')
    await flushPromises()
    expect(writeText).toHaveBeenCalledWith('cmake ..\nmake -j')

    const pre = log.element
    Object.defineProperty(pre, 'scrollHeight', { configurable: true, value: 400 })
    Object.defineProperty(pre, 'clientHeight', { configurable: true, value: 100 })
    pre.scrollTop = 300
    await log.trigger('scroll')
    expect(wrapper.get('.task-toast__log-follow').attributes('aria-pressed')).toBe('true')

    pre.scrollTop = 0
    await log.trigger('scroll')
    expect(wrapper.get('.task-toast__log-follow').attributes('aria-pressed')).toBe('false')

    await wrapper.get('.task-toast__log-follow').trigger('click')
    expect(wrapper.get('.task-toast__log-follow').attributes('aria-pressed')).toBe('true')
  })

  it('keeps following when new log lines grow the view without the user scrolling up', async () => {
    seedTask({
      task_id: 'build',
      type: 'build',
      status: 'running',
      progress: 40,
      description: 'Building 1Cat-vLLM',
    })
    const store = useProgressStore()
    store.taskLogs = { build: ['cmake ..'] }

    const wrapper = mountTray()
    await flushPromises()
    await wrapper.get('.task-toast__body').trigger('click')
    await flushPromises()

    const pre = wrapper.get('.task-toast__logs').element
    Object.defineProperty(pre, 'clientHeight', { configurable: true, value: 100 })
    Object.defineProperty(pre, 'scrollHeight', { configurable: true, writable: true, value: 400 })
    pre.scrollTop = 300
    await wrapper.get('.task-toast__logs').trigger('scroll')
    expect(wrapper.get('.task-toast__log-follow').attributes('aria-pressed')).toBe('true')

    pre.scrollHeight = 900
    await wrapper.get('.task-toast__logs').trigger('scroll')
    expect(wrapper.get('.task-toast__log-follow').attributes('aria-pressed')).toBe('true')
    expect(pre.scrollTop).toBe(900)
  })
})

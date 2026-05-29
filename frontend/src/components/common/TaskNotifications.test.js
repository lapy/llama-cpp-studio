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
    setActivePinia(createPinia())
    axios.post.mockReset()
    axios.post.mockResolvedValue({ data: { ok: true, message: 'Stopped' } })
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
    })

    const wrapper = mountTray()
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading model.gguf')
    expect(wrapper.text()).toContain('42%')
  })

  it('opens detail dialog when a toast is clicked', async () => {
    seedTask({
      task_id: 'build',
      type: 'build',
      status: 'running',
      progress: 10,
      description: 'Building llama.cpp',
    })

    const wrapper = mountTray()
    await flushPromises()

    await wrapper.find('.task-toast__body').trigger('click')
    await flushPromises()

    expect(wrapper.find('.detail-panel-stub').exists()).toBe(true)
    expect(wrapper.find('.detail-panel-stub').text()).toBe('build')
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

    await wrapper.find('.task-toast__dismiss').trigger('click')
    await flushPromises()

    expect(store.getTask('done')).toBeNull()
    expect(wrapper.find('.task-toast').exists()).toBe(false)
  })

  it.each(REAL_TASK_FIXTURES)(
    'requests cancellation via $cancelEndpoint for $label',
    async ({ task, cancelEndpoint }) => {
      seedTask(task)
      const wrapper = mountTray()
      await flushPromises()

      await wrapper.find('.task-toast__stop').trigger('click')
      await flushPromises()

      expect(axios.post).toHaveBeenCalledWith(cancelEndpoint, { task_id: task.task_id })
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

    expect(wrapper.find('.task-toast__stop').exists()).toBe(false)
  })
})

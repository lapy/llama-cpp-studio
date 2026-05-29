import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import axios from 'axios'
import TaskDetailPanel from './TaskDetailPanel.vue'
import { useProgressStore } from '@/stores/progress'
import { REAL_TASK_FIXTURES } from '@/test-fixtures/taskFixtures.js'

vi.mock('axios', () => ({
  default: { post: vi.fn() },
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: vi.fn() }),
}))

describe('TaskDetailPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    axios.post.mockReset()
    axios.post.mockResolvedValue({ data: { ok: true } })
  })

  function mountPanel(props = {}) {
    return mount(TaskDetailPanel, {
      props: {
        showCompleted: true,
        dismissible: true,
        ...props,
      },
      global: {
        stubs: {
          Button: {
            props: ['label', 'loading', 'disabled'],
            emits: ['click'],
            template: '<button :disabled="disabled" @click="$emit(\'click\', { stopPropagation() {} })">{{ label }}</button>',
          },
        },
      },
    })
  }

  it('filters tasks by task-id prop', async () => {
    const store = useProgressStore()
    store.tasks = {
      a: { task_id: 'a', type: 'build', status: 'running', progress: 5, description: 'Build A' },
      b: { task_id: 'b', type: 'download', status: 'running', progress: 10, description: 'Download B' },
    }

    const wrapper = mountPanel({ taskId: 'a' })
    await flushPromises()

    expect(wrapper.text()).toContain('Build A')
    expect(wrapper.text()).not.toContain('Download B')
  })

  it('filters install tasks by metadata.manager', async () => {
    const store = useProgressStore()
    store.tasks = {
      cuda: {
        task_id: 'cuda',
        type: 'install',
        status: 'running',
        progress: 5,
        description: 'CUDA',
        metadata: { manager: 'cuda' },
      },
      lm: {
        task_id: 'lm',
        type: 'install',
        status: 'running',
        progress: 5,
        description: 'LMDeploy',
        metadata: { manager: 'lmdeploy' },
      },
    }

    const wrapper = mountPanel({
      type: 'install',
      metadataKey: 'manager',
      metadataValue: 'cuda',
    })
    await flushPromises()

    expect(wrapper.text()).toContain('CUDA')
    expect(wrapper.text()).not.toContain('LMDeploy')
  })

  it('posts to the correct cancel endpoint when stop is clicked', async () => {
    const fixture = REAL_TASK_FIXTURES.find((f) => f.label === 'lmdeploy install')
    const store = useProgressStore()
    store.tasks = { [fixture.task.task_id]: fixture.task }

    const wrapper = mountPanel({ taskId: fixture.task.task_id })
    await flushPromises()

    const stopBtn = wrapper.findAll('button').find((b) => b.text() === 'Stop')
    await stopBtn.trigger('click')
    await flushPromises()

    expect(axios.post).toHaveBeenCalledWith(fixture.cancelEndpoint, {
      task_id: fixture.task.task_id,
    })
  })

  it('does not show stop for completed tasks', async () => {
    const store = useProgressStore()
    store.tasks = {
      done: {
        task_id: 'done',
        type: 'build',
        status: 'completed',
        progress: 100,
        description: 'Done',
      },
    }

    const wrapper = mountPanel({ taskId: 'done' })
    await flushPromises()

    expect(wrapper.text()).not.toContain('Stop')
  })

  it('shows logs when toggled', async () => {
    const store = useProgressStore()
    store.tasks = {
      build_1: {
        task_id: 'build_1',
        type: 'build',
        status: 'running',
        progress: 50,
        description: 'Build',
      },
    }
    store.taskLogs = { build_1: ['cmake ..', 'make -j'] }

    const wrapper = mountPanel({ taskId: 'build_1' })
    await flushPromises()

    await wrapper.find('.logs-toggle').trigger('click')
    await flushPromises()

    expect(wrapper.find('.task-logs').text()).toContain('cmake ..')
    expect(wrapper.find('.task-logs').text()).toContain('make -j')
  })

  it('dismisses a finished task from the panel', async () => {
    const store = useProgressStore()
    store.tasks = {
      done: {
        task_id: 'done',
        type: 'download',
        status: 'completed',
        progress: 100,
        description: 'Done',
      },
    }

    const wrapper = mountPanel({ taskId: 'done' })
    await flushPromises()

    await wrapper.find('.dismiss-task-btn').trigger('click')
    await flushPromises()

    expect(store.getTask('done')).toBeNull()
  })
})

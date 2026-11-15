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
    localStorage.removeItem('llama-studio.activity.dismissed')
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
            template:
              '<button :disabled="disabled" @click="$emit(\'click\', { stopPropagation() {} })">{{ label }}</button>',
          },
        },
      },
    })
  }

  it('filters tasks by task-id prop', async () => {
    const store = useProgressStore()
    store.tasks = {
      a: {
        task_id: 'a',
        type: 'build',
        status: 'running',
        progress: 5,
        description: 'Build A',
      },
      b: {
        task_id: 'b',
        type: 'download',
        status: 'running',
        progress: 10,
        description: 'Download B',
      },
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

  it('shows aggregate model bytes and remaining size', async () => {
    const store = useProgressStore()
    store.tasks = {
      download: {
        task_id: 'download',
        type: 'download',
        status: 'running',
        progress: 25,
        description: 'Download model',
        metadata: {
          bytes_downloaded: 250_000_000,
          total_bytes: 1_000_000_000,
          files_completed: 1,
          files_total: 4,
        },
      },
    }

    const wrapper = mountPanel({ taskId: 'download' })
    await flushPromises()

    expect(wrapper.find('.download-progress-meta').text()).toBe(
      '250 MB / 1.0 GB · 750 MB left · file 1/4',
    )
  })

  it('shows the CUDA toolkit a native build used', async () => {
    const store = useProgressStore()
    store.tasks = {
      build_cuda: {
        task_id: 'build_cuda',
        type: 'build',
        status: 'completed',
        progress: 100,
        description: 'Build llama.cpp main',
        metadata: {
          cuda: {
            cuda_home: '/app/data/cuda/cuda-12.8.2',
            cuda_version: '12.8.2',
            source: 'studio-managed',
          },
        },
      },
    }

    const wrapper = mountPanel({ taskId: 'build_cuda' })
    await flushPromises()

    expect(wrapper.find('.cuda-provenance').text()).toBe(
      'Built with CUDA 12.8.2 (/app/data/cuda/cuda-12.8.2, studio-managed)',
    )
    expect(wrapper.text()).not.toContain('CUDA: not recorded')
  })

  it('shows CUDA: not recorded for a build without provenance', async () => {
    const store = useProgressStore()
    store.tasks = {
      build_legacy: {
        task_id: 'build_legacy',
        type: 'build',
        status: 'completed',
        progress: 100,
        description: 'Build llama.cpp main',
        metadata: { version_name: 'source-main-1700000000' },
      },
    }

    const wrapper = mountPanel({ taskId: 'build_legacy' })
    await flushPromises()

    expect(wrapper.find('.cuda-provenance--missing').text()).toBe('CUDA: not recorded')
  })

  it('does not show a CUDA row on non-build tasks', async () => {
    const store = useProgressStore()
    store.tasks = {
      download_1: {
        task_id: 'download_1',
        type: 'download',
        status: 'completed',
        progress: 100,
        description: 'Download model',
      },
    }

    const wrapper = mountPanel({ taskId: 'download_1' })
    await flushPromises()

    expect(wrapper.find('.cuda-provenance').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('CUDA')
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

  it('copies task logs when Copy logs is clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const store = useProgressStore()
    store.tasks = {
      scan_1: {
        task_id: 'scan_1',
        type: 'param_scan',
        status: 'completed',
        progress: 100,
        description: 'Scan llama.cpp CLI parameters',
      },
    }
    store.taskLogs = { scan_1: ['===== HELP OUTPUT =====', 'EXTRACT accept key=ctx_size'] }

    const wrapper = mountPanel({ taskId: 'scan_1' })
    await flushPromises()

    await wrapper.find('.logs-toggle').trigger('click')
    await flushPromises()
    const copyBtn = wrapper.find('button[aria-label="Copy"]')
    expect(copyBtn.exists()).toBe(true)
    await copyBtn.trigger('click')
    await flushPromises()

    expect(writeText).toHaveBeenCalledWith('===== HELP OUTPUT =====\nEXTRACT accept key=ctx_size')
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

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import axios from 'axios'
import NeedsAttentionPanel from './NeedsAttentionPanel.vue'
import { useProgressStore } from '@/stores/progress'

const { toastAdd } = vi.hoisted(() => ({ toastAdd: vi.fn() }))

vi.mock('axios', () => ({
  default: { get: vi.fn(), post: vi.fn() },
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

const UPDATED_AT = 1_760_000_000

function unknownRow(overrides = {}) {
  return {
    operation_id: 'build-unknown',
    task_id: 'build-unknown',
    kind: 'build',
    status: 'unknown',
    description: 'Build llama.cpp',
    message: 'Whether this action finished could not be established.',
    resource_key: 'engine:demo',
    evidence: 'unproven',
    retry: 'withheld',
    needs_confirmation: true,
    updated_at: UPDATED_AT,
    actions: ['view_logs'],
    recovery: null,
    ...overrides,
  }
}

function safeRow(overrides = {}) {
  return {
    operation_id: 'build-not-started',
    task_id: 'build-not-started',
    kind: 'build',
    status: 'interrupted',
    description: 'Build llama.cpp (never started)',
    message: 'The action had not started. It was not run again.',
    resource_key: 'engine:demo',
    evidence: 'negative',
    retry: 'safe',
    needs_confirmation: false,
    updated_at: UPDATED_AT - 10,
    actions: ['view_logs', 'dismiss'],
    recovery: null,
    ...overrides,
  }
}

function mockAttention(operations) {
  vi.mocked(axios.get).mockResolvedValue({
    data: { operations, count: operations.length },
  })
}

function mountPanel() {
  return mount(NeedsAttentionPanel, {
    global: {
      stubs: {
        Button: {
          props: ['label', 'disabled', 'loading'],
          template: '<button :disabled="disabled">{{ label }}</button>',
        },
        LoadingState: true,
        LogOutput: {
          props: ['lines', 'loading', 'emptyMessage'],
          template: '<div class="log-output-stub">{{ (lines || []).join("\\n") }}</div>',
        },
      },
    },
  })
}

function buttonByText(wrapper, text) {
  return wrapper.findAll('button').find((button) => button.text().includes(text))
}

describe('NeedsAttentionPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    toastAdd.mockReset()
    vi.mocked(axios.get).mockReset()
  })

  it('renders one page section with a refresh action', async () => {
    mockAttention([])

    const wrapper = mountPanel()
    await flushPromises()

    const section = wrapper.get('#attention')
    expect(section.classes()).toContain('page-section')
    expect(section.get('.page-section__title').text()).toBe('Operations needing attention')
    expect(section.attributes('aria-busy')).toBe('false')
    const refresh = section
      .get('.page-section__actions')
      .findAll('button')
      .find((button) => button.text() === 'Refresh')
    expect(refresh).toBeTruthy()
    await refresh.trigger('click')
    await flushPromises()
    expect(axios.get).toHaveBeenCalledTimes(2)
  })

  it('renders status, reason, timestamp and only backend-supported actions', async () => {
    mockAttention([unknownRow(), safeRow()])

    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.get('h2').text()).toBe('Operations needing attention')
    expect(wrapper.text()).toContain('Build llama.cpp')
    expect(wrapper.text()).toContain('Outcome unknown')
    expect(wrapper.text()).toContain('Whether this action finished could not be established.')
    expect(wrapper.text()).toContain('Interrupted')
    const timestamp = wrapper.find('time')
    expect(timestamp.attributes('datetime')).toBe(new Date(UPDATED_AT * 1000).toISOString())

    const labels = wrapper.findAll('button').map((button) => button.text())
    expect(labels.some((label) => label.includes('View logs'))).toBe(true)
    expect(labels.some((label) => label.includes('Acknowledge'))).toBe(true)
    expect(labels.some((label) => /retry|confirm/i.test(label))).toBe(false)
    expect(wrapper.find('.attention-item__guidance').text()).toContain(
      'Studio will not run this again on its own.',
    )
  })

  it('shows a quiet success state when nothing needs attention', async () => {
    mockAttention([])

    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.text()).toContain('No interrupted operations.')
    const banner = wrapper.get('.state-banner--success')
    expect(banner.attributes('role')).toBe('status')
    expect(wrapper.findAll('.attention-item')).toHaveLength(0)
  })

  it('shows a readable error with a retry action instead of an empty state', async () => {
    vi.mocked(axios.get).mockRejectedValue({
      response: { data: { detail: 'Operation records could not be read.' } },
    })

    const wrapper = mountPanel()
    await flushPromises()

    const banner = wrapper.get('[role="alert"]')
    expect(banner.get('.state-banner__message').text()).toBe('Operation records could not be read.')
    expect(wrapper.text()).not.toContain('No interrupted operations.')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'Could not check interrupted operations',
      }),
    )

    mockAttention([safeRow()])
    await buttonByText(banner, 'Retry').trigger('click')
    await flushPromises()

    expect(axios.get).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('Build llama.cpp (never started)')
  })

  it('loads retained logs for one operation when View logs is clicked', async () => {
    const row = unknownRow()
    vi.mocked(axios.get).mockImplementation((url) => {
      if (url === '/api/operations/attention') {
        return Promise.resolve({ data: { operations: [row], count: 1 } })
      }
      if (url === '/api/operations/tasks/build-unknown/logs') {
        return Promise.resolve({ data: { lines: ['cmake output', 'compiler stopped'] } })
      }
      return Promise.reject(new Error(`unexpected request ${url}`))
    })

    const wrapper = mountPanel()
    await flushPromises()
    const toggle = buttonByText(wrapper, 'View logs')
    expect(toggle.attributes('aria-expanded')).toBe('false')

    await toggle.trigger('click')
    await flushPromises()

    expect(axios.get).toHaveBeenCalledWith('/api/operations/tasks/build-unknown/logs')
    expect(wrapper.find('.log-output-stub').text()).toBe('cmake output\ncompiler stopped')
    expect(buttonByText(wrapper, 'Hide logs').attributes('aria-expanded')).toBe('true')

    await buttonByText(wrapper, 'Hide logs').trigger('click')
    await flushPromises()

    expect(wrapper.find('.log-output-stub').exists()).toBe(false)
  })

  it('dismisses only never-started work through the existing dismiss endpoint', async () => {
    mockAttention([unknownRow(), safeRow()])
    vi.mocked(axios.post).mockResolvedValue({ data: { ok: true } })
    const store = useProgressStore()

    const wrapper = mountPanel()
    await flushPromises()

    // The unknown row keeps its admission fence: only the safe row offers it.
    expect(wrapper.findAll('.attention-action--quiet')).toHaveLength(1)
    await wrapper.get('.attention-action--quiet').trigger('click')
    await flushPromises()

    expect(axios.post).toHaveBeenCalledWith('/api/tasks/dismiss', {
      task_id: 'build-not-started',
    })
    expect(store.attentionOperations.map((row) => row.operation_id)).toEqual(['build-unknown'])
    expect(wrapper.text()).not.toContain('Build llama.cpp (never started)')
    expect(toastAdd).toHaveBeenCalledWith(expect.objectContaining({ severity: 'info' }))
  })

  it('keeps the row and reports failure when acknowledgement is refused', async () => {
    mockAttention([safeRow()])
    vi.mocked(axios.post).mockRejectedValue({
      response: { data: { detail: 'Task is still running' } },
    })
    const store = useProgressStore()

    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.get('.attention-action--quiet').trigger('click')
    await flushPromises()

    expect(store.attentionOperations).toHaveLength(1)
    expect(wrapper.text()).toContain('Build llama.cpp (never started)')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        detail: 'Task is still running',
      }),
    )
  })
})

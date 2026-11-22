import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axios from 'axios'
import ProxyView from './ProxyView.vue'

vi.mock('axios', () => ({
  default: { get: vi.fn(), put: vi.fn(), post: vi.fn(), patch: vi.fn(), delete: vi.fn() },
}))

const confirmRequire = vi.fn((options) => options.accept?.())
const toastAdd = vi.fn()
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: confirmRequire }) }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('@/composables/singleConfirm', () => ({
  requireSingleConfirmation: (confirm, options) => confirm.require(options),
}))

function statusPayload() {
  return {
    settings: {
      backend: 'studio',
      proxy_port: 2000,
      routing_aliases: { fast: ['demo-a', 'demo-b'] },
      default_model: null,
      idle_ttl_seconds: null,
      max_body_bytes: null,
      allow_anonymous: true,
      max_loaded_models: null,
      global_concurrency: null,
      per_model_concurrency: null,
      queue_timeout_seconds: 30,
      request_timeout_seconds: null,
      readiness_timeout_seconds: 2700,
      drain_timeout_seconds: 30,
      vram_admission: 'warn',
      vram_safety_margin_mib: 1024,
      vram_estimates: { demo: 8192 },
    },
    configured_backend: 'studio',
    active_backend: 'studio',
    proxy_running: true,
    running: [
      {
        model: 'demo',
        engine: 'llama.cpp',
        pid: 42,
        port: 3010,
        ready: true,
        state: 'ready',
        inflight: 2,
        created_at: 1,
        vram: {
          observed_mib: 8000,
          estimate_mib: 8192,
          estimate_source: 'settings',
          quality: 'observed',
        },
      },
    ],
    activity: [
      { type: 'starting', model: 'demo', created_at: 1, pid: 42 },
      { type: 'ready', model: 'demo', created_at: 2, pid: 42 },
      {
        type: 'vram_warning',
        model: 'demo',
        reason: 'Estimate 16000 MiB (file_size) exceeds 7168 MiB available',
        decision: 'warned',
        created_at: 3,
      },
    ],
    vram: {
      mode: 'warn',
      safety_margin_mib: 1024,
      estimates_configured: { demo: 8192 },
      observed_estimates: { demo: { mib: 8000, at: '2026-01-01T00:00:00Z', devices: [0] } },
      probe: { status: 'ok', vendor: 'nvidia', devices: [{ index: 0, free_mib: 8192 }] },
      reservations: [],
      last_admission: {
        decision: 'warned',
        model: 'demo',
        estimate_mib: 16000,
        estimate_source: 'file_size',
        available_mib: 7168,
        deficit_mib: 8832,
        reason: 'Estimate 16000 MiB (file_size) exceeds 7168 MiB available; warn mode admits.',
      },
    },
    catalog: [
      { id: 'demo', studio_model_id: 'stored-demo', engine: 'llama.cpp', aliases: ['demo-lite'] },
    ],
    network: {
      listen_host: '127.0.0.1',
      port: 2000,
      listen_source: 'loopback default',
      tls_note: 'plain HTTP',
    },
  }
}

function mountView() {
  return mount(ProxyView, {
    attachTo: document.body,
    global: {
      stubs: {
        RouterLink: {
          name: 'RouterLink',
          props: ['to'],
          template: '<a class="model-link"><slot /></a>',
        },
        Dialog: {
          props: ['visible', 'header'],
          template:
            '<div v-if="visible" class="dialog-stub" :data-header="header"><slot /><slot name="footer" /></div>',
        },
        Button: {
          props: ['label', 'disabled'],
          emits: ['click'],
          template: '<button :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>',
        },
        Tag: { props: ['value', 'severity'], template: '<span>{{ value }}</span>' },
        Message: { template: '<div><slot /></div>' },
        ProgressBar: {
          props: ['value'],
          template: '<div class="progress-stub" :data-value="value" />',
        },
        ToggleSwitch: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :checked="modelValue" @change="$emit(\'update:modelValue\', $event.target.checked)" />',
        },
        PageHeader: {
          props: ['title'],
          template: '<div><slot name="meta" /><slot name="actions" /><slot /></div>',
        },
        EmptyState: {
          props: ['title'],
          template: '<div class="empty-stub">{{ title }}<slot /></div>',
        },
        LoadingState: {
          props: ['message'],
          template: '<div class="loading-stub">{{ message }}</div>',
        },
      },
    },
  })
}

async function clickButton(wrapper, label) {
  const button = wrapper.findAll('button').find((candidate) => candidate.text() === label)
  expect(button, `button "${label}"`).toBeTruthy()
  expect(button.attributes('disabled')).toBeUndefined()
  await button.trigger('click')
  await flushPromises()
}

describe('ProxyView', () => {
  beforeEach(() => {
    vi.mocked(axios.get).mockClear()
    vi.mocked(axios.patch).mockReset()
    vi.mocked(axios.post).mockClear()
    vi.mocked(axios.put).mockClear()
    vi.mocked(axios.delete).mockClear()
    confirmRequire.mockClear()
    confirmRequire.mockImplementation((options) => options.accept?.())
    toastAdd.mockClear()
    vi.mocked(axios.get).mockImplementation((url) => {
      if (String(url).includes('/logs'))
        return Promise.resolve({ data: { model: 'demo', lines: ['stderr: boom'] } })
      if (String(url).includes('keys')) {
        return Promise.resolve({
          data: {
            keys: [
              {
                id: 'key-1',
                name: 'Production',
                prefix: 'sk-studio-test',
                models: [],
                scopes: ['inference'],
              },
            ],
          },
        })
      }
      return Promise.resolve({ data: statusPayload() })
    })
    vi.mocked(axios.post).mockResolvedValue({ data: {} })
    vi.mocked(axios.put).mockResolvedValue({ data: { restart_required: true } })
  })

  it('keeps unsaved settings through manual refresh', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Settings & routing')
    const port = wrapper.findAll('input[type="number"]')[0]
    await port.setValue('2333')
    await wrapper.get('button[aria-label="Refresh proxy status"]').trigger('click')
    await flushPromises()
    expect(port.element.value).toBe('2333')
    expect(
      wrapper
        .findAll('button')
        .find((button) => button.text() === 'Save settings')
        .attributes('disabled'),
    ).toBeUndefined()
    wrapper.unmount()
  })

  it('coalesces overlapping status reads and aborts on unmount', async () => {
    let complete
    let signal
    axios.get.mockImplementation((url, options) => {
      if (url === '/api/proxy') {
        signal = options.signal
        return new Promise((resolve) => {
          complete = resolve
        })
      }
      return Promise.resolve({ data: { keys: [] } })
    })
    const wrapper = mountView()
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.loading-stub').text()).toBe('Loading proxy state…')
    await wrapper.get('button[aria-label="Refresh proxy status"]').trigger('click')
    expect(axios.get.mock.calls.filter(([url]) => url === '/api/proxy')).toHaveLength(1)
    wrapper.unmount()
    expect(signal.aborted).toBe(true)
    complete({ data: statusPayload() })
    await flushPromises()
  })

  it('backs off failed status reads while preserving the last good runtime data', async () => {
    vi.useFakeTimers()
    let wrapper
    try {
      wrapper = mountView()
      await flushPromises()
      axios.get.mockImplementation((url) =>
        url === '/api/proxy'
          ? Promise.reject(new Error('Connection unavailable'))
          : Promise.resolve({ data: { keys: [] } }),
      )
      const reads = () => axios.get.mock.calls.filter(([url]) => url === '/api/proxy').length
      await vi.advanceTimersByTimeAsync(5000)
      expect(reads()).toBe(2)
      expect(wrapper.text()).toContain('demo')
      expect(wrapper.text()).toContain('Connection interrupted')
      expect(wrapper.get('.state-banner--error').text()).toContain('Connection unavailable')
      expect(wrapper.get('.state-banner--error').text()).toContain('Retry')
      await vi.advanceTimersByTimeAsync(9999)
      expect(reads()).toBe(2)
      await vi.advanceTimersByTimeAsync(1)
      expect(reads()).toBe(3)
      await vi.advanceTimersByTimeAsync(19999)
      expect(reads()).toBe(3)
    } finally {
      wrapper?.unmount()
      vi.useRealTimers()
    }
  })

  it('surfaces status, keys, runtime rows, network, and activity', async () => {
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('Proxy active · studio')
    expect(wrapper.text()).toContain('Production')
    expect(wrapper.text()).toContain('demo')
    expect(wrapper.text()).toContain('127.0.0.1')
    // Activity renders newest-first with human text, not raw JSON.
    const activity = wrapper.find('.activity-list')
    expect(activity.text()).toContain('ready')
    expect(activity.text().indexOf('ready')).toBeLessThan(activity.text().indexOf('starting'))
    expect(activity.text()).not.toContain('{')
    wrapper.unmount()
  })

  it('keeps model lifecycle management out of proxy diagnostics', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.findAll('[role="tab"]').map((tab) => tab.text())).not.toContain('Models')
    for (const label of ['Preload', 'Warmup', 'Reload', 'Unload', 'Remove']) {
      expect(wrapper.findAll('button').some((button) => button.text() === label)).toBe(false)
    }
    expect(wrapper.find('.runtime .model-link').text()).toContain('Manage model')
    expect(wrapper.find('.runtime').findComponent({ name: 'RouterLink' }).props('to')).toEqual({
      name: 'model-config',
      params: { id: 'stored-demo' },
    })
    await clickButton(wrapper, 'View logs')
    expect(wrapper.find('.log-body').text()).toContain('boom')
    expect(wrapper.find('.log-body').text()).not.toContain('stderr:')
    wrapper.unmount()
  })

  it('admits only one key creation while a save is pending', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'New key')
    await wrapper
      .findAll('label')
      .find((label) => label.text().trim() === 'Key name')
      .get('input')
      .setValue('Client')
    let finish
    vi.mocked(axios.post).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        }),
    )
    const button = wrapper.findAll('button').find((candidate) => candidate.text() === 'Create key')
    const first = button.trigger('click')
    const second = button.trigger('click')
    await Promise.all([first, second])
    expect(axios.post).toHaveBeenCalledTimes(1)
    finish({ data: { key: 'sk-studio-once' } })
    await flushPromises()
    expect(wrapper.get('#inference-key-secret').element.value).toBe('sk-studio-once')
    wrapper.unmount()
  })

  it('creates token-budgeted keys and edits policies without rotating secrets', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'New key')
    async function setField(label, value, last = false) {
      const labels = wrapper.findAll('label').filter((item) => item.text().trim() === label)
      await labels[last ? labels.length - 1 : 0].find('input').setValue(value)
    }
    await setField('Key name', 'Automation')
    await setField('Daily token budget (UTC)', '10000')
    await setField('Maximum output tokens per request', '64')
    await clickButton(wrapper, 'Create key')
    expect(axios.post).toHaveBeenCalledWith(
      '/api/inference/keys',
      expect.objectContaining({
        name: 'Automation',
        daily_token_budget: 10000,
        max_output_tokens: 64,
      }),
    )
    await clickButton(wrapper, 'Edit policy')
    await setField('Daily token budget (UTC)', '20000', true)
    vi.mocked(axios.patch).mockResolvedValue({ data: {} })
    await clickButton(wrapper, 'Save policy')
    expect(axios.patch).toHaveBeenCalledWith(
      '/api/inference/keys/key-1',
      expect.objectContaining({ daily_token_budget: 20000 }),
    )
    expect(axios.post).not.toHaveBeenCalledWith('/api/inference/keys/key-1/rotate')
    wrapper.unmount()
  })

  it('filters and pages request history, showing unknown usage explicitly', async () => {
    const wrapper = mountView()
    await flushPromises()
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({
        data: String(url).includes('/requests')
          ? {
              total: 250,
              requests: [
                {
                  id: 'trace',
                  state: 'cancelled',
                  status: 499,
                  model: 'demo',
                  key_name: 'Production',
                  input_tokens: null,
                  output_tokens: null,
                  usage_source: 'reservation',
                },
              ],
            }
          : statusPayload(),
      }),
    )
    await wrapper
      .findAll('[role="tab"]')
      .find((tab) => tab.text() === 'Requests')
      .trigger('click')
    await flushPromises()
    await wrapper.find('.request-filters input').setValue('trace')
    await clickButton(wrapper, 'Apply')
    expect(axios.get).toHaveBeenCalledWith('/api/proxy/requests', {
      params: { limit: 100, offset: 0, search: 'trace' },
      signal: expect.any(AbortSignal),
    })
    expect(wrapper.find('tbody').text()).toContain('cancelled')
    expect(wrapper.find('tbody').text()).toContain('reservation')
    expect(wrapper.find('tbody').text()).toContain('— / —')
    await clickButton(wrapper, 'Next 100')
    expect(axios.get).toHaveBeenCalledWith('/api/proxy/requests', {
      params: { limit: 100, offset: 100, search: 'trace' },
      signal: expect.any(AbortSignal),
    })
    wrapper.unmount()
  })

  it('submits request filters from the form and resets them to the unfiltered first page', async () => {
    const wrapper = mountView()
    await flushPromises()
    await wrapper
      .findAll('[role="tab"]')
      .find((tab) => tab.text() === 'Requests')
      .trigger('click')
    await flushPromises()

    const requestReads = () => axios.get.mock.calls.filter(([url]) => url === '/api/proxy/requests')
    await wrapper.get('#request-filter-search').setValue('trace')
    await wrapper.get('#request-filter-state').setValue('failed')
    // Enter in any field submits the form; the request toolbar uses the same path.
    await wrapper.get('.request-filters').trigger('submit')
    await flushPromises()
    expect(requestReads().at(-1)[1].params).toEqual({
      limit: 100,
      offset: 0,
      search: 'trace',
      state: 'failed',
    })

    await clickButton(wrapper, 'Reset')
    expect(requestReads().at(-1)[1].params).toEqual({ limit: 100, offset: 0 })
    expect(wrapper.get('#request-filter-search').element.value).toBe('')
    expect(wrapper.get('#request-filter-state').element.value).toBe('')
    wrapper.unmount()
  })

  it('keeps the policy editor open on validation failure', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Edit policy')
    vi.mocked(axios.patch).mockRejectedValue({
      response: { data: { detail: 'Budget must be positive' } },
    })
    await clickButton(wrapper, 'Save policy')
    expect(wrapper.text()).toContain('Budget must be positive')
    expect(wrapper.findAll('button').some((button) => button.text() === 'Save policy')).toBe(true)
    wrapper.unmount()
  })

  it('shows failed engines as failures without counting them as running', async () => {
    const payload = statusPayload()
    payload.running[0] = {
      ...payload.running[0],
      state: 'failed',
      ready: false,
      failure_reason: 'Engine exited unexpectedly (code 137)',
    }
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({ data: String(url).includes('keys') ? { keys: [] } : payload }),
    )
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.text()).toContain('0 running')
    expect(wrapper.text()).not.toContain('1 running')
    expect(wrapper.text()).toContain('Engine exited unexpectedly (code 137)')
    expect(wrapper.find('.runtime .model-link').text()).toContain('Manage model')
    wrapper.unmount()
  })

  it('removes a failed engine instance and clears its card status', async () => {
    const failed = statusPayload()
    failed.running[0] = {
      ...failed.running[0],
      state: 'failed',
      ready: false,
      failure_reason: 'Engine exited unexpectedly (code 137)',
    }
    let removed = false
    vi.mocked(axios.get).mockImplementation((url) => {
      if (String(url).includes('keys')) return Promise.resolve({ data: { keys: [] } })
      return Promise.resolve({ data: removed ? statusPayload() : failed })
    })
    vi.mocked(axios.delete).mockImplementation(async () => {
      removed = true
      return { status: 204 }
    })

    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Remove')
    await flushPromises()

    expect(axios.delete).toHaveBeenCalledWith('/api/proxy/models/demo/instance')
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ summary: 'Failed engine removed' }),
    )
    expect(wrapper.text()).not.toContain('Engine exited unexpectedly')
    expect(wrapper.findAll('button').some((button) => button.text() === 'Remove')).toBe(false)
    wrapper.unmount()
  })

  it('does not silently turn a zero budget into unlimited access', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'New key')
    const labels = wrapper.findAll('.key-options label')
    await labels
      .find((label) => label.text().trim() === 'Key name')
      .find('input')
      .setValue('Strict')
    await labels
      .find((label) => label.text().trim() === 'Daily token budget (UTC)')
      .find('input')
      .setValue('0')
    await wrapper.find('.key-options select').setValue('exact')
    vi.mocked(axios.post).mockRejectedValue({
      response: { data: { detail: 'daily_token_budget must be positive' } },
    })
    await clickButton(wrapper, 'Create key')
    expect(axios.post).toHaveBeenCalledWith(
      '/api/inference/keys',
      expect.objectContaining({ daily_token_budget: 0, token_accounting: 'exact' }),
    )
    expect(wrapper.text()).toContain('daily_token_budget must be positive')
    wrapper.unmount()
  })

  it('offers an explicit banner action when the proxy is down', async () => {
    const payload = statusPayload()
    payload.active_backend = ''
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }),
    )
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('The proxy is stopped')
    expect(wrapper.get('.state-banner--warning').text()).toContain('The proxy is stopped')
    await clickButton(wrapper, 'Start proxy')
    expect(axios.post).toHaveBeenCalledWith('/api/proxy/lifecycle/start')
    wrapper.unmount()
  })

  it('persists the memory-only request payload capture setting', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Settings & routing')

    const toggle = wrapper.get('[data-capture-request-payloads]')
    expect(toggle.element.checked).toBe(false)
    await toggle.setValue(true)
    await clickButton(wrapper, 'Save settings')
    await flushPromises()

    const [, body] = vi.mocked(axios.put).mock.calls[0]
    expect(body.capture_request_payloads).toBe(true)
    wrapper.unmount()
  })

  it('disables save until the form is dirty and saves native access settings', async () => {
    const wrapper = mountView()
    await flushPromises()

    const save = () => wrapper.findAll('button').find((b) => b.text() === 'Save settings')
    expect(save().attributes('disabled')).toBeDefined()

    await wrapper.find('.inline-control input[type="checkbox"]').setValue(false)
    await flushPromises()
    expect(save().attributes('disabled')).toBeUndefined()

    await clickButton(wrapper, 'Save settings')
    const [, body] = vi.mocked(axios.put).mock.calls[0]
    expect(body.allow_anonymous).toBe(false)
    expect(body.routing_aliases).toEqual({ fast: ['demo-a', 'demo-b'] })
    expect(wrapper.text()).toContain('Restart & apply')
    wrapper.unmount()
  })

  it('keeps native routing aliases without offering an external backend switch', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.text()).not.toContain('Switch backend')
    expect(wrapper.text()).toContain('Add alias')
    expect(wrapper.find('select option[value="native_proxy"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('rotates keys behind a confirmation', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { key: 'sk-studio-rotated' } })
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Rotate')

    expect(confirmRequire).toHaveBeenCalled()
    expect(axios.post).toHaveBeenCalledWith('/api/inference/keys/key-1/rotate')
    expect(wrapper.get('#inference-key-secret').element.value).toBe('sk-studio-rotated')
    expect(wrapper.text()).toContain('the old one no longer works')
    wrapper.unmount()
  })

  it('does not act when a confirmation is dismissed', async () => {
    confirmRequire.mockImplementation(() => false)
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Revoke')
    expect(axios.delete).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('filters engine logs in place through the compact toolbar', async () => {
    vi.mocked(axios.get).mockImplementation((url) => {
      if (String(url).includes('/logs')) {
        return Promise.resolve({
          data: {
            model: 'demo',
            sequence: 3,
            lines: ['stdout: listening', 'stderr: boom error', 'stdout: WARNING slow'],
          },
        })
      }
      if (String(url).includes('keys')) return Promise.resolve({ data: { keys: [] } })
      return Promise.resolve({ data: statusPayload() })
    })
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'View logs')

    expect(axios.get).toHaveBeenCalledWith('/api/proxy/models/demo/logs')
    const logReads = () =>
      axios.get.mock.calls.filter(([url]) => url === '/api/proxy/models/demo/logs').length
    expect(logReads()).toBe(1)

    await wrapper.find('.log-search input').setValue('boom')
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(1)
    expect(wrapper.find('.log-body').text()).toContain('boom')

    await wrapper.find('.log-search input').setValue('')
    await wrapper.find('.log-level').trigger('click') // All
    await wrapper.findAll('.log-level')[2].trigger('click') // Error
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(1)
    expect(wrapper.find('.log-body').text()).toContain('boom')

    await wrapper.findAll('.log-level')[1].trigger('click') // Warn
    await flushPromises()
    expect(wrapper.findAll('.log-line')).toHaveLength(2)
    expect(logReads()).toBe(1)
    wrapper.unmount()
  })

  it('shows loading progress for cold-start generations', async () => {
    const payload = statusPayload()
    payload.running[0] = {
      ...payload.running[0],
      ready: false,
      state: 'loading',
      loading_progress: 42.5,
    }
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }),
    )
    const wrapper = mountView()
    await flushPromises()

    const bar = wrapper.find('.progress-stub')
    expect(bar.exists()).toBe(true)
    expect(bar.attributes('data-value')).toBe('43')
    wrapper.unmount()
  })

  it('renders one section tablist with roving keyboard selection and wired panels', async () => {
    const wrapper = mountView()
    await flushPromises()

    const tablist = wrapper.get('[role="tablist"]')
    expect(tablist.attributes('aria-label')).toBe('Proxy sections')
    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs).toHaveLength(5)
    expect(tabs[0].attributes('aria-selected')).toBe('true')
    expect(tabs[0].attributes('aria-controls')).toBe('proxy-panel-overview')
    expect(wrapper.get('[role="tabpanel"]').attributes('id')).toBe('proxy-panel-overview')

    tabs[0].element.focus()
    await tabs[0].trigger('keydown', { key: 'ArrowRight' })
    await flushPromises()
    expect(tabs[1].attributes('aria-selected')).toBe('true')
    expect(document.activeElement).toBe(tabs[1].element)
    expect(wrapper.get('[role="tabpanel"]').attributes('id')).toBe('proxy-panel-requests')
    expect(wrapper.get('[role="tabpanel"]').attributes('aria-labelledby')).toBe(
      'proxy-tab-requests',
    )

    await tabs[1].trigger('keydown', { key: 'End' })
    expect(tabs[4].attributes('aria-selected')).toBe('true')
    wrapper.unmount()
  })

  it('renders every proxy tab as shared sections with page headings', async () => {
    const wrapper = mountView()
    await flushPromises()

    // Overview cards and the fleet live inside shared PageSection surfaces.
    expect(wrapper.findAll('.page-section').length).toBeGreaterThanOrEqual(2)
    expect(wrapper.get('#proxy-traffic-title').text()).toBe('Recent traffic')

    await clickButton(wrapper, 'Settings & routing')
    expect(wrapper.get('#proxy-vram-title').text()).toBe('GPU admission')
    expect(wrapper.get('#proxy-network-title').text()).toBe('Network')
    const headings = wrapper.findAll('h2').map((node) => node.text())
    expect(headings).toContain('Serving limits')
    expect(headings).toContain('Routing')

    await clickButton(wrapper, 'Access keys')
    expect(wrapper.get('.key-list').text()).toContain('Production')
    expect(wrapper.get('.key-row').text()).toContain('sk-studio-test')
    wrapper.unmount()
  })

  it('states an empty activity log with a next step instead of a blank panel', async () => {
    const payload = statusPayload()
    payload.activity = []
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({ data: String(url).includes('keys') ? { keys: [] } : payload }),
    )
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Events')

    const empty = wrapper
      .findAll('.empty-stub')
      .find((node) => node.text().includes('No proxy activity'))
    expect(empty).toBeTruthy()
    wrapper.unmount()
  })

  it('keeps request empty state inside the shared table shell', async () => {
    const payload = statusPayload()
    payload.requests = []
    payload.request_total = 0
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({
        data: String(url).includes('/requests')
          ? { total: 0, requests: [] }
          : String(url).includes('keys')
            ? { keys: [] }
            : payload,
      }),
    )
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Requests')

    expect(wrapper.find('.table-shell').exists()).toBe(true)
    expect(wrapper.find('.table-shell__scroll').exists()).toBe(true)
    expect(wrapper.find('.table-shell__state').text()).toContain('No requests recorded yet')
    expect(wrapper.findAll('button').some((button) => button.text() === 'Refresh')).toBe(true)
    wrapper.unmount()
  })
})

describe('Proxy monitoring', () => {
  it('renders request rows and opens their details', async () => {
    const payload = statusPayload()
    payload.requests = [
      {
        id: 'r1',
        completed_at: 123,
        model: 'demo',
        method: 'POST',
        path: '/v1/chat/completions',
        status: 200,
        latency_ms: 12,
        output_tokens: 4,
      },
    ]
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({ data: String(url).includes('keys') ? { keys: [] } : payload }),
    )
    const wrapper = mountView()
    await flushPromises()
    const tab = wrapper.findAll('[role="tab"]').find((tab) => tab.text() === 'Requests')
    await tab.trigger('click')
    expect(tab.attributes('aria-selected')).toBe('true')
    expect(wrapper.find('tbody').text()).toContain('/v1/chat/completions')
    await clickButton(wrapper, 'Details')
    expect(wrapper.find('.request-inspector').text()).toContain('12')
    expect(wrapper.find('.request-inspector').text()).not.toContain('Time to first byte')
    await wrapper.get('.inspector-field-toggle input').setValue(true)
    expect(wrapper.find('.request-inspector').text()).toContain('Time to first byte')
    await clickButton(wrapper, 'Reset stats')
    expect(axios.post).toHaveBeenCalledWith('/api/proxy/stats/reset')
    wrapper.unmount()
  })

  it('shows GPU admission evidence, the overview warning, and running VRAM', async () => {
    const wrapper = mountView()
    await flushPromises()
    // Overview is the default tab and carries the admission warning.
    expect(wrapper.find('.vram-notice').text()).toContain('demo needs 16,000 MiB file_size')
    expect(wrapper.find('.vram-notice').text()).toContain('7,168 MiB available')
    expect(wrapper.text()).toContain('8,000 MiB observed')
    await clickButton(wrapper, 'Settings & routing')
    expect(wrapper.get('#proxy-vram-title').text()).toBe('GPU admission')
    expect(wrapper.text()).toContain('Probe ready')
    expect(wrapper.text()).toContain('demo · 8,000 MiB')
    wrapper.unmount()
  })

  it('saves live GPU admission settings and per-model estimates', async () => {
    const payload = statusPayload()
    payload.catalog = [
      ...payload.catalog,
      { id: 'other', studio_model_id: 'stored-other', engine: 'llama.cpp', aliases: [] },
    ]
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({ data: String(url).includes('keys') ? { keys: [] } : payload }),
    )
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Settings & routing')
    await wrapper.get('select[aria-label="VRAM admission mode"]').setValue('enforce')
    await wrapper.get('input[aria-label="VRAM safety margin MiB"]').setValue('2048')
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Add estimate')
      .trigger('click')
    await flushPromises()
    await wrapper.get('#vram-model-1').setValue('other')
    await wrapper.get('#vram-mib-1').setValue('4096')
    await clickButton(wrapper, 'Save settings')
    expect(axios.put).toHaveBeenCalledWith(
      '/api/proxy/settings',
      expect.objectContaining({
        vram_admission: 'enforce',
        vram_safety_margin_mib: 2048,
        vram_estimates: { demo: 8192, other: 4096 },
      }),
    )
    wrapper.unmount()
  })

  it('surfaces VRAM admission activity in the event log', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Events')
    const activity = wrapper.find('.activity-list')
    expect(activity.text()).toContain('vram_warning')
    expect(activity.text()).toContain('exceeds 7168 MiB available')
    wrapper.unmount()
  })

  it('labels a buffered response where TTFT is not observable', async () => {
    const payload = statusPayload()
    payload.requests = [
      {
        id: 'buffered',
        completed_at: 123,
        model: 'demo',
        method: 'POST',
        path: '/v1/chat/completions',
        state: 'completed',
        status: 200,
        requested_stream: false,
        response_transport: 'buffered',
        nonstream_response: true,
        incremental_output: false,
        ttft_ms: null,
        latency_ms: 2249.7,
        input_tokens: 7687,
        input_tokens_cached: 4736,
        output_tokens: 152,
      },
    ]
    vi.mocked(axios.get).mockImplementation((url) =>
      Promise.resolve({ data: String(url).includes('keys') ? { keys: [] } : payload }),
    )
    const wrapper = mountView()
    await flushPromises()
    const tab = wrapper.findAll('[role="tab"]').find((tab) => tab.text() === 'Requests')
    await tab.trigger('click')
    const rowText = wrapper.find('tbody').text()
    expect(rowText).toContain('non-stream / 2249.7 ms')
    await clickButton(wrapper, 'Details')
    expect(wrapper.find('.request-inspector').text()).toContain('Response transport')
    wrapper.unmount()
  })
})

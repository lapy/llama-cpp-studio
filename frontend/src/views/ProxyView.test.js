import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axios from 'axios'
import ProxyView from './ProxyView.vue'

vi.mock('axios', () => ({
  default: { get: vi.fn(), put: vi.fn(), post: vi.fn(), delete: vi.fn() },
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
      backend: 'studio', proxy_port: 2000, routing_aliases: { fast: ['demo-a', 'demo-b'] },
      default_model: null, idle_ttl_seconds: null, max_body_bytes: null, allow_anonymous: true,
      max_loaded_models: null, global_concurrency: null, per_model_concurrency: null,
      queue_timeout_seconds: 30, request_timeout_seconds: null,
      readiness_timeout_seconds: 2700, drain_timeout_seconds: 30,
    },
    configured_backend: 'studio', active_backend: 'studio', proxy_running: true,
    running: [{ model: 'demo', engine: 'llama.cpp', pid: 42, port: 3010, ready: true, state: 'ready', inflight: 2, created_at: 1 }],
    activity: [{ type: 'starting', model: 'demo', created_at: 1, pid: 42 }, { type: 'ready', model: 'demo', created_at: 2, pid: 42 }],
    catalog: [{ id: 'demo', engine: 'llama.cpp', aliases: ['demo-lite'] }],
    network: { listen_host: '127.0.0.1', port: 2000, listen_source: 'loopback default', tls_note: 'plain HTTP' },
  }
}

function mountView() {
  return mount(ProxyView, {
    global: {
      stubs: {
        Button: { props: ['label', 'disabled'], emits: ['click'], template: '<button :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>' },
        Tag: { props: ['value', 'severity'], template: '<span>{{ value }}</span>' },
        Message: { template: '<div><slot /></div>' },
        ProgressBar: { props: ['value'], template: '<div class="progress-stub" :data-value="value" />' },
        ToggleSwitch: { props: ['modelValue'], emits: ['update:modelValue'], template: '<input type="checkbox" :checked="modelValue" @change="$emit(\'update:modelValue\', $event.target.checked)" />' },
        PageHeader: { props: ['title'], template: '<div><slot name="meta" /><slot name="actions" /><slot /></div>' },
        EmptyState: { props: ['title'], template: '<div class="empty-stub">{{ title }}<slot /></div>' },
        LoadingState: { props: ['message'], template: '<div class="loading-stub">{{ message }}</div>' },
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
    vi.mocked(axios.post).mockClear()
    vi.mocked(axios.put).mockClear()
    vi.mocked(axios.delete).mockClear()
    confirmRequire.mockClear()
    confirmRequire.mockImplementation((options) => options.accept?.())
    toastAdd.mockClear()
    vi.mocked(axios.get).mockImplementation((url) => {
      if (String(url).includes('/logs')) return Promise.resolve({ data: { model: 'demo', lines: ['stderr: boom'] } })
      if (String(url).includes('keys')) {
        return Promise.resolve({ data: { keys: [{ id: 'key-1', name: 'Production', prefix: 'sk-studio-test', models: [], scopes: ['inference'] }] } })
      }
      return Promise.resolve({ data: statusPayload() })
    })
    vi.mocked(axios.post).mockResolvedValue({ data: {} })
    vi.mocked(axios.put).mockResolvedValue({ data: { restart_required: true } })
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

  it('offers an explicit banner action when the proxy is down', async () => {
    const payload = statusPayload()
    payload.active_backend = ''
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }))
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('is not running')
    await clickButton(wrapper, 'Start proxy')
    expect(axios.post).toHaveBeenCalledWith('/api/proxy/lifecycle/start')
    wrapper.unmount()
  })

  it('switches to the persisted configuration, not an unsaved form selection', async () => {
    // Persisted config says llama_swap; the proxy actively running is studio.
    const payload = statusPayload()
    payload.settings.backend = 'llama_swap'
    payload.configured_backend = 'llama_swap'
    payload.active_backend = 'studio'
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }))
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('The active proxy is “studio”')
    // Operator changes the dropdown before switching — the switch must still
    // honour the saved configuration the banner describes.
    await wrapper.find('select').setValue('studio')
    await clickButton(wrapper, 'Switch backend')
    expect(axios.post).toHaveBeenCalledWith('/api/proxy/switch', { backend: 'llama_swap' })
    wrapper.unmount()
  })

  it('surfaces a toast when an engine action fails', async () => {
    vi.mocked(axios.post).mockImplementation((url) =>
      String(url).includes('/reload')
        ? Promise.reject({ response: { data: { detail: 'engine crashed' } } })
        : Promise.resolve({ data: {} }))
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Reload')

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'error', detail: 'engine crashed' }),
    )
    // The page must not end up in a permanently broken state.
    expect(wrapper.text()).toContain('Proxy active · studio')
    wrapper.unmount()
  })

  it('disables save until the form is dirty and sends aliases as lists', async () => {
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

  it('rejects duplicate alias names client-side', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Add alias')
    const aliasInputs = wrapper.findAll('.alias-row input')
    await aliasInputs[2].setValue('FAST')
    await aliasInputs[3].setValue('demo')
    await clickButton(wrapper, 'Save settings')

    expect(axios.put).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('Alias names must be unique')
    wrapper.unmount()
  })

  it('rotates keys behind a confirmation', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { key: 'sk-studio-rotated' } })
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Rotate')

    expect(confirmRequire).toHaveBeenCalled()
    expect(axios.post).toHaveBeenCalledWith('/api/inference/keys/key-1/rotate')
    expect(wrapper.text()).toContain('sk-studio-rotated')
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

  it('preloads and warms catalog models', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { model: 'demo', ready: true, warmed: true } })
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Warmup')

    expect(axios.post).toHaveBeenCalledWith('/api/proxy/models/demo/preload', null, { params: { warm: true } })
    wrapper.unmount()
  })

  it('filters engine logs through search and severity controls', async () => {
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'View logs')

    expect(axios.get).toHaveBeenCalledWith('/api/proxy/models/demo/logs', { params: {} })
    await wrapper.find('.log-search input').setValue('error')
    await vi.waitFor(() => {
      expect(axios.get).toHaveBeenCalledWith('/api/proxy/models/demo/logs', {
        params: { search: 'error', level: undefined },
      })
    })
    await wrapper.find('.log-toolbar select').setValue('error')
    await flushPromises()
    expect(axios.get).toHaveBeenCalledWith('/api/proxy/models/demo/logs', {
      params: { search: 'error', level: 'error' },
    })
    wrapper.unmount()
  })

  it('unloads an idle engine without asking', async () => {
    const payload = statusPayload()
    payload.running[0].inflight = 0
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }))
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Unload')

    expect(confirmRequire).not.toHaveBeenCalled()
    expect(axios.post).toHaveBeenCalledWith('/api/proxy/models/demo/unload')
    wrapper.unmount()
  })

  it('confirms unload when requests are in flight', async () => {
    const payload = statusPayload()
    payload.running[0].inflight = 3
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }))
    confirmRequire.mockImplementation(() => false)
    const wrapper = mountView()
    await flushPromises()
    await clickButton(wrapper, 'Unload')

    expect(confirmRequire).toHaveBeenCalled()
    expect(axios.post).not.toHaveBeenCalledWith('/api/proxy/models/demo/unload')
    wrapper.unmount()
  })

  it('shows loading progress for cold-start generations', async () => {
    const payload = statusPayload()
    payload.running[0] = { ...payload.running[0], ready: false, state: 'loading', loading_progress: 42.5 }
    vi.mocked(axios.get).mockImplementation((url) =>
      String(url).includes('keys')
        ? Promise.resolve({ data: { keys: [] } })
        : Promise.resolve({ data: payload }))
    const wrapper = mountView()
    await flushPromises()

    const bar = wrapper.find('.progress-stub')
    expect(bar.exists()).toBe(true)
    expect(bar.attributes('data-value')).toBe('43')
    wrapper.unmount()
  })
})

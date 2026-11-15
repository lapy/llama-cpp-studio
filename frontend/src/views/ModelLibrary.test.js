import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import ModelLibrary from './ModelLibrary.vue'
import { LIBRARY_PRESETS_KEY } from '@/utils/libraryFilterPresets'

const fetchModels = vi.fn().mockResolvedValue(undefined)
const fetchSafetensorsModels = vi.fn().mockResolvedValue(undefined)
const fetchHuggingfaceTokenStatus = vi.fn().mockResolvedValue(undefined)
const toastAdd = vi.fn()

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
  RouterLink: {
    name: 'RouterLink',
    props: ['to'],
    template: '<a :href="to"><slot /></a>',
  },
}))

const confirmRequire = vi.fn()
vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: (...args) => confirmRequire(...args) }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: (...args) => toastAdd(...args) }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    llamaVersions: [],
    ikLlamaVersions: [],
    unslothLlamaVersions: [],
    lmdeployVersions: [],
    onecatVllmVersions: [],
    sglangVersions: [],
    sglangV100Versions: [],
    vllmVersions: [],
    audioCppVersions: [],
    systemStatus: {},
  }),
}))

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => ({
    subscribe: () => () => {},
    subscribeToDownloadComplete: () => () => {},
  }),
}))

const modelStore = reactive({
  models: [],
  loading: false,
  safetensorsLoading: false,
  hasHuggingfaceToken: true,
  modelsError: null,
  modelsStale: false,
  allQuantizations: [],
  fetchModels,
  fetchSafetensorsModels,
  fetchHuggingfaceTokenStatus,
  startModel: vi.fn(),
  stopModel: vi.fn(),
})

vi.mock('@/stores/models', () => ({
  useModelStore: () => modelStore,
}))

async function flushWithFakeTimers() {
  const pending = flushPromises()
  await vi.advanceTimersByTimeAsync(0)
  await pending
}

function mountLibrary() {
  return mount(ModelLibrary, {
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label', 'disabled', 'loading', 'ariaLabel', 'icon', 'text', 'severity', 'size'],
          emits: ['click'],
          template:
            '<button :aria-label="ariaLabel" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
        },
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        Dialog: true,
        Toast: true,
        Password: true,
        ConfirmDialog: true,
        PageHeader: { template: '<div><slot name="meta" /><slot name="actions" /></div>' },
        LoadingState: true,
        EmptyState: {
          props: ['title'],
          template: '<div class="empty-stub">{{ title }}<slot /></div>',
        },
        ModelRow: {
          props: ['quant'],
          template:
            '<div class="quant-row" :class="{ \'is-active\': quant.is_active }">{{ quant.quantization }}</div>',
        },
        ModelWarmupButton: true,
      },
    },
  })
}

describe('ModelLibrary running glow', () => {
  beforeEach(() => {
    vi.useRealTimers()
    fetchModels.mockReset()
    fetchModels.mockResolvedValue(undefined)
    fetchSafetensorsModels.mockReset()
    fetchSafetensorsModels.mockResolvedValue(undefined)
    modelStore.loading = false
    modelStore.safetensorsLoading = false
    modelStore.hasHuggingfaceToken = true
    modelStore.modelsError = null
    modelStore.modelsStale = false
    modelStore.models = []
    modelStore.startModel.mockReset()
    modelStore.stopModel.mockReset()
    toastAdd.mockReset()
    localStorage.removeItem(LIBRARY_PRESETS_KEY)
    localStorage.removeItem('llama-studio.library.view')
  })

  it('marks a group running when a quantization is active', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/idle',
        quantizations: [{ id: 'idle', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
      {
        huggingface_id: 'org/live',
        quantizations: [
          { id: 'q4', quantization: 'Q4_K_M', is_active: false, status: 'stopped' },
          { id: 'q8', quantization: 'Q8_0', is_active: true, status: 'ready' },
        ],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    const groups = wrapper.findAll('.model-group')
    expect(groups).toHaveLength(2)
    expect(groups[0].classes()).not.toContain('is-running')
    expect(groups[1].classes()).toContain('is-running')
    wrapper.unmount()
  })

  it('marks a group running while a model is loading even if not yet active', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/loading',
        quantizations: [{ id: 'm1', quantization: 'Q4', is_active: false, status: 'loading' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    expect(wrapper.get('.model-group').classes()).toContain('is-running')
    wrapper.unmount()
  })

  it('does not glow idle groups', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/idle',
        quantizations: [{ id: 'm1', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    expect(wrapper.get('.model-group').classes()).not.toContain('is-running')
    wrapper.unmount()
  })

  it('does not reschedule the catalog poll after unmount during a delayed refresh', async () => {
    vi.useFakeTimers()
    fetchModels.mockResolvedValue(undefined)
    const wrapper = mountLibrary()
    await flushWithFakeTimers()
    expect(fetchSafetensorsModels).not.toHaveBeenCalled()

    let release
    fetchModels.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve
        }),
    )
    await vi.advanceTimersByTimeAsync(5000)
    expect(fetchModels).toHaveBeenCalledTimes(2)
    wrapper.unmount()
    release()
    await flushWithFakeTimers()
    await vi.advanceTimersByTimeAsync(15000)
    expect(fetchModels).toHaveBeenCalledTimes(2)
    vi.useRealTimers()
  })

  it('pauses routine polls while hidden and refreshes once when shown again', async () => {
    vi.useFakeTimers()
    let visibility = 'visible'
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => visibility,
    })
    fetchModels.mockResolvedValue(undefined)
    const wrapper = mountLibrary()
    await flushWithFakeTimers()
    const afterMount = fetchModels.mock.calls.length

    visibility = 'hidden'
    document.dispatchEvent(new Event('visibilitychange'))
    await vi.advanceTimersByTimeAsync(15000)
    expect(fetchModels.mock.calls.length).toBe(afterMount)

    visibility = 'visible'
    document.dispatchEvent(new Event('visibilitychange'))
    await flushWithFakeTimers()
    expect(fetchModels.mock.calls.length).toBe(afterMount + 1)
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('shows an unreachable proxy instead of a verified stopped library', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        quantizations: [
          {
            id: 'q4',
            quantization: 'Q4_K_M',
            is_active: false,
            status: null,
            runtime_quality: 'unreachable',
          },
        ],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    expect(wrapper.text()).toContain('not shown as stopped')
    expect(wrapper.get('.model-group').classes()).not.toContain('is-running')
    wrapper.unmount()
  })

  it('shows a failed library request instead of an empty library', async () => {
    modelStore.models = []
    modelStore.loading = false
    modelStore.modelsError = 'Service unavailable'
    const wrapper = mountLibrary()
    await flushPromises()
    expect(wrapper.text()).toContain('Could not load models')
    expect(wrapper.text()).not.toContain('Add your first model')
    wrapper.unmount()
  })

  it('shows first-model onboarding with starter search deep links when the library is empty', async () => {
    const wrapper = mountLibrary()
    await flushPromises()

    const panel = wrapper.get('.first-model')
    expect(panel.text()).toContain('Add your first model')
    expect(panel.text()).toContain('Starter models')

    const searchLink = wrapper
      .findAll('.first-model a')
      .find((link) => link.text().trim() === 'Search models')
    expect(searchLink).toBeTruthy()
    expect(searchLink.attributes('href')).toBe('/search')

    const cards = wrapper.findAll('.first-model__list li .first-model__card')
    expect(cards).toHaveLength(3)
    expect(cards.map((card) => card.attributes('href'))).toEqual([
      '/search?q=Qwen%2FQwen2.5-0.5B-Instruct-GGUF',
      '/search?q=Qwen%2FQwen2.5-1.5B-Instruct-GGUF',
      '/search?q=bartowski%2FLlama-3.2-1B-Instruct-GGUF',
    ])
    expect(cards[0].text()).toContain('Qwen2.5 0.5B Instruct')
    expect(cards[0].text()).toContain('Qwen/Qwen2.5-0.5B-Instruct-GGUF')
    expect(cards[0].text()).toContain('fastest first download')
    wrapper.unmount()
  })

  it('hides the first-model empty state once the library has a model', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()

    expect(wrapper.find('.first-model').exists()).toBe(false)
    expect(wrapper.find('.model-group').exists()).toBe(true)
    wrapper.unmount()
  })

  it('confirms and retries a start withheld by unresolved prior work', async () => {
    vi.useFakeTimers()
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        quantizations: [
          {
            id: 'q4',
            quantization: 'Q4_K_M',
            format: 'safetensors',
            is_active: false,
            status: 'stopped',
          },
        ],
      },
    ]
    modelStore.startModel
      .mockReset()
      .mockRejectedValueOnce({
        response: {
          data: {
            detail: {
              code: 'ACTION_RETRY_WITHHELD',
              operation_id: 'start-unknown',
              state_token: 'tok-1',
              message:
                'Prior work may already have happened. Confirm this operation and its current state before trying again.',
            },
          },
        },
      })
      .mockResolvedValue({ model: 'q4', state: 'loading' })
    confirmRequire.mockReset().mockImplementation((options) => options.accept())

    const wrapper = mountLibrary()
    await flushWithFakeTimers()

    await wrapper.get('button[aria-label="Start org/live"]').trigger('click')
    await flushWithFakeTimers()

    expect(modelStore.startModel).toHaveBeenCalledTimes(2)
    expect(modelStore.startModel).toHaveBeenNthCalledWith(1, 'q4', null)
    expect(modelStore.startModel).toHaveBeenNthCalledWith(2, 'q4', {
      confirm_operation_id: 'start-unknown',
      confirm_state: 'tok-1',
    })
    expect(confirmRequire).toHaveBeenCalledWith(
      expect.objectContaining({ header: 'Confirm model start' }),
    )
    wrapper.unmount()
    vi.useRealTimers()
  })

  it('shows inline start progress, a cancel Stop and Starting on the card while a start is in flight', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        base_model_name: 'Live model',
        quantizations: [{ id: 'q4', quantization: 'Q4_K_M', is_active: false, status: 'stopped' }],
      },
    ]
    let resolveStart
    modelStore.startModel.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveStart = resolve
        }),
    )
    const wrapper = mountLibrary()
    await flushPromises()

    await wrapper.get('button[aria-label="Start Live model"]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Starting Q4_K_M…')
    expect(wrapper.get('.running-badge').text()).toBe('Starting')
    expect(
      wrapper.get('button[aria-label="Stop Live model"]').attributes('disabled'),
    ).toBeUndefined()

    modelStore.models[0].quantizations[0].is_active = true
    modelStore.models[0].quantizations[0].status = 'ready'
    resolveStart({})
    await flushPromises()

    expect(wrapper.text()).not.toContain('Starting Q4_K_M…')
    expect(wrapper.get('button[aria-label="Connect Live model"]')).toBeTruthy()
    wrapper.unmount()
  })

  it('keeps a failed start actionable with Retry and surfaces last_error', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        base_model_name: 'Live model',
        quantizations: [
          {
            id: 'q4',
            quantization: 'Q4_K_M',
            format: 'safetensors',
            is_active: false,
            status: 'stopped',
          },
        ],
      },
    ]
    modelStore.startModel.mockRejectedValueOnce(new Error('CUDA out of memory'))
    const wrapper = mountLibrary()
    await flushPromises()

    fetchModels.mockImplementation(async () => {
      modelStore.models[0].quantizations[0].status = 'failed'
      modelStore.models[0].quantizations[0].last_error =
        'CUDA out of memory while loading 7B weights'
    })

    await wrapper.get('button[aria-label="Start Live model"]').trigger('click')
    await flushPromises()
    await flushPromises()

    const retry = wrapper.get('button[aria-label="Retry start for Live model"]')
    expect(retry.attributes('disabled')).toBeUndefined()
    expect(wrapper.get('.model-actions__error').text()).toBe(
      'CUDA out of memory while loading 7B weights',
    )
    expect(wrapper.get('.model-actions__error').attributes('title')).toBe(
      'CUDA out of memory while loading 7B weights',
    )
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        group: 'model-library',
        configureModelId: 'q4',
      }),
    )

    await retry.trigger('click')
    await flushPromises()
    expect(modelStore.startModel).toHaveBeenCalledTimes(2)
    wrapper.unmount()
  })

  it('refreshes the library after a successful start so the card becomes Connect', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        base_model_name: 'Live model',
        quantizations: [
          {
            id: 'q4',
            quantization: 'Q4_K_M',
            format: 'safetensors',
            is_active: false,
            status: 'stopped',
          },
        ],
      },
    ]
    modelStore.startModel.mockImplementation(async () => {
      modelStore.models[0].quantizations[0].is_active = true
      modelStore.models[0].quantizations[0].status = 'ready'
    })
    const wrapper = mountLibrary()
    await flushPromises()

    await wrapper.get('button[aria-label="Start Live model"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('button[aria-label="Connect Live model"]')).toBeTruthy()
    expect(wrapper.findAll('button[aria-label="Start Live model"]')).toHaveLength(0)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'Model started' }),
    )
    wrapper.unmount()
  })

  it('offers Refresh status for a stale model and reloads the library from it', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/live',
        base_model_name: 'Live model',
        quantizations: [
          {
            id: 'q4',
            quantization: 'Q4_K_M',
            is_active: false,
            status: 'ready',
            runtime_quality: 'stale',
          },
        ],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    expect(wrapper.get('.running-badge').text()).toBe('Status unavailable')
    fetchModels.mockClear()

    await wrapper.get('button[aria-label="Refresh status for Live model"]').trigger('click')
    await flushPromises()

    expect(fetchModels).toHaveBeenCalled()
    wrapper.unmount()
  })

  it('selects visible models, shows the indeterminate state and clears on filter changes', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        base_model_name: 'Alpha',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
      {
        huggingface_id: 'org/b',
        base_model_name: 'Beta',
        quantizations: [{ id: 'b', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()

    const selectAll = wrapper.get('input[aria-label="Select all visible models"]')
    await selectAll.setValue(true)
    await flushPromises()
    expect(wrapper.get('input[aria-label="Select Alpha"]').element.checked).toBe(true)
    expect(wrapper.get('input[aria-label="Select Beta"]').element.checked).toBe(true)
    expect(wrapper.get('.library-bulk').text()).toContain('2 selected')

    await wrapper.get('input[aria-label="Select Beta"]').setValue(false)
    await flushPromises()
    expect(wrapper.get('input[aria-label="Select all visible models"]').element.indeterminate).toBe(
      true,
    )
    expect(wrapper.get('.library-bulk').text()).toContain('1 selected')

    await wrapper.get('#library-status').setValue('stopped')
    await flushPromises()
    const clearedAll = wrapper.get('input[aria-label="Select all visible models"]')
    expect(clearedAll.element.checked).toBe(false)
    expect(clearedAll.element.indeterminate).toBe(false)
    expect(wrapper.get('input[aria-label="Select Alpha"]').element.checked).toBe(false)
    expect(wrapper.find('.library-bulk').exists()).toBe(false)
    wrapper.unmount()
  })

  it('drops selected models that disappear from the visible library', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        base_model_name: 'Alpha',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
      {
        huggingface_id: 'org/b',
        base_model_name: 'Beta',
        quantizations: [{ id: 'b', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()
    await wrapper.get('input[aria-label="Select all visible models"]').setValue(true)
    await flushPromises()

    modelStore.models = modelStore.models.slice(1)
    await flushPromises()

    expect(wrapper.find('input[aria-label="Select Alpha"]').exists()).toBe(false)
    expect(wrapper.get('input[aria-label="Select Beta"]').element.checked).toBe(true)
    expect(wrapper.get('.library-bulk').text()).toContain('1 selected')
    wrapper.unmount()
  })

  it('selects visible rows in the list layout', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        base_model_name: 'Alpha',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
      {
        huggingface_id: 'org/b',
        base_model_name: 'Beta',
        quantizations: [{ id: 'b', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    const wrapper = mountLibrary()
    await flushPromises()

    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'List')
      .trigger('click')
    await flushPromises()

    expect(wrapper.findAll('.library-list__row')).toHaveLength(2)
    await wrapper.get('input[aria-label="Select all visible models"]').setValue(true)
    await flushPromises()
    expect(wrapper.get('input[aria-label="Select Alpha"]').element.checked).toBe(true)
    expect(wrapper.get('input[aria-label="Select Beta"]').element.checked).toBe(true)
    expect(wrapper.get('.library-bulk').text()).toContain('2 selected')
    wrapper.unmount()
  })

  it('runs bulk start through the store and clears selection after the summary', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        base_model_name: 'Alpha',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
      {
        huggingface_id: 'org/b',
        base_model_name: 'Beta',
        quantizations: [{ id: 'b', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    modelStore.startModel.mockResolvedValue(undefined)
    const wrapper = mountLibrary()
    await flushPromises()
    await wrapper.get('input[aria-label="Select all visible models"]').setValue(true)
    await flushPromises()

    const startButton = wrapper
      .findAll('button')
      .find((button) => button.text() === 'Start selected')
    expect(startButton).toBeTruthy()
    await startButton.trigger('click')
    await flushPromises()
    await flushPromises()

    expect(modelStore.startModel.mock.calls.map(([modelId]) => modelId)).toEqual(['a', 'b'])
    expect(wrapper.get('.library-bulk').text()).toContain('Started 2 of 2 selected models')
    expect(wrapper.get('input[aria-label="Select Alpha"]').element.checked).toBe(false)
    expect(wrapper.get('input[aria-label="Select Beta"]').element.checked).toBe(false)
    wrapper.unmount()
  })

  it('applies a saved preset to the same filters and URL-backed controls', async () => {
    modelStore.models = [
      {
        huggingface_id: 'org/a',
        base_model_name: 'Alpha',
        quantizations: [{ id: 'a', quantization: 'Q4', is_active: false, status: 'stopped' }],
      },
    ]
    localStorage.setItem(
      LIBRARY_PRESETS_KEY,
      JSON.stringify({
        version: 1,
        presets: [
          {
            id: 'preset-focused',
            name: 'Focused',
            filters: {
              q: 'alpha',
              status: 'running',
              engine: 'all',
              task: 'all',
              sort: 'size',
            },
            createdAt: '2026-11-15T00:00:00.000Z',
          },
        ],
      }),
    )
    const wrapper = mountLibrary()
    await flushPromises()

    await wrapper.get('button[aria-label="Apply preset Focused"]').trigger('click')
    await flushPromises()

    expect(wrapper.get('#library-search').element.value).toBe('alpha')
    expect(wrapper.get('#library-status').element.value).toBe('running')
    expect(wrapper.get('#library-sort').element.value).toBe('size')
    wrapper.unmount()
  })

  it('applies a success glow to running groups and active quants', () => {
    const src = readFileSync(
      resolve(dirname(fileURLToPath(import.meta.url)), './ModelLibrary.vue'),
      'utf8',
    )
    expect(src).toContain('.model-group.is-running')
    expect(src).toContain('var(--glow-success)')
    expect(src).toContain(':deep(.quant-row.is-active)')
  })
})

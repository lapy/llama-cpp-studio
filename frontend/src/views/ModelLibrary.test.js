import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import ModelLibrary from './ModelLibrary.vue'

const fetchModels = vi.fn().mockResolvedValue(undefined)
const fetchSafetensorsModels = vi.fn().mockResolvedValue(undefined)
const fetchHuggingfaceTokenStatus = vi.fn().mockResolvedValue(undefined)

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: vi.fn() }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: vi.fn() }),
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

function mountLibrary() {
  return mount(ModelLibrary, {
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: true,
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
        Dialog: true,
        Password: true,
        ConfirmDialog: true,
        PageHeader: { template: '<div><slot name="meta" /><slot name="actions" /></div>' },
        LoadingState: true,
        EmptyState: true,
        ModelRow: {
          props: ['quant'],
          template: '<div class="quant-row" :class="{ \'is-active\': quant.is_active }">{{ quant.quantization }}</div>',
        },
        ModelStartStopButton: true,
      },
    },
  })
}

describe('ModelLibrary running glow', () => {
  beforeEach(() => {
    fetchModels.mockClear()
    modelStore.loading = false
    modelStore.safetensorsLoading = false
    modelStore.hasHuggingfaceToken = true
    modelStore.models = []
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

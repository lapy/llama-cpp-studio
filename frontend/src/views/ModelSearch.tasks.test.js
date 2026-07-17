import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { reactive, ref } from 'vue'
import axios from 'axios'
import ModelSearch from './ModelSearch.vue'
import { useProgressStore } from '@/stores/progress'

const downloadModel = vi.fn()
const downloadGgufBundle = vi.fn()
const downloadSafetensorsBundle = vi.fn()
const fetchModels = vi.fn()
const fetchSafetensorsModels = vi.fn()
const fetchHuggingfaceTokenStatus = vi.fn().mockResolvedValue(undefined)

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn().mockResolvedValue(undefined) }),
  useRoute: () => ({ name: 'search', query: {} }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: vi.fn() }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    engineDescriptors: [],
    fetchEngineDescriptors: vi.fn().mockResolvedValue(undefined),
  }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => modelStore,
}))

const modelStore = reactive({
  models: [],
  safetensorsModels: [],
  hasHuggingfaceToken: true,
  searchQuery: ref(''),
  searchLastQuery: ref(''),
  searchHasSearched: ref(true),
  searchResults: ref([]),
  searchLoading: ref(false),
  searchFormat: ref('gguf'),
  catalogFacets: ref({}),
  catalogProviderStatus: ref({}),
  catalogTotal: ref(0),
  catalogPage: ref(1),
  catalogHasMore: ref(false),
  allQuantizations: [],
  downloadModel,
  downloadGgufBundle,
  downloadSafetensorsBundle,
  fetchModels,
  fetchSafetensorsModels,
  fetchHuggingfaceTokenStatus,
})

function mountModelSearch() {
  return mount(ModelSearch, {
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: {
          props: ['label', 'loading'],
          template: '<button :data-label="label" :data-loading="loading ? \'1\' : \'0\'" @click="$emit(\'click\')">{{ label }}</button>',
        },
        Tag: {
          props: ['value'],
          template: '<span class="tag">{{ value }}</span>',
        },
        InputText: true,
        Dropdown: true,
        LoadingState: true,
        EmptyState: true,
      },
    },
  })
}

describe('ModelSearch task integration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    downloadModel.mockReset()
    downloadGgufBundle.mockReset()
    downloadSafetensorsBundle.mockReset()
    fetchModels.mockReset()
    fetchSafetensorsModels.mockReset()
    axios.get.mockReset()
    axios.get.mockResolvedValue({ data: { sizes: {} } })
    downloadGgufBundle.mockImplementation(() => new Promise(() => {}))
    downloadModel.mockImplementation(() => new Promise(() => {}))
    downloadSafetensorsBundle.mockImplementation(() => new Promise(() => {}))
    fetchModels.mockResolvedValue(undefined)
    fetchSafetensorsModels.mockResolvedValue(undefined)

    modelStore.searchFormat = 'gguf'
    modelStore.searchResults = [
      {
        id: 'org/model',
        modelId: 'org/model',
        pipeline_tag: 'text-generation',
        quantizations: {
          Q4_K_M: {
            quantization: 'Q4_K_M',
            files: [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
          },
        },
      },
    ]
  })

  async function expandAndGetDownloadButton(wrapper) {
    await wrapper.find('.result-main').trigger('click')
    await flushPromises()
    const downloadBtn = wrapper.findAll('button').find((btn) => btn.text() === 'Download')
    expect(downloadBtn).toBeTruthy()
    return downloadBtn
  }

  it('clears optimistic pending state when matching task_created arrives', async () => {
    const wrapper = mountModelSearch()
    await flushPromises()
    const progressStore = useProgressStore()

    const downloadBtn = await expandAndGetDownloadButton(wrapper)
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading')

    progressStore.handleEvent('task_created', {
      task_id: 'download_gguf_bundle_org_model_Q4_K_M_1',
      type: 'download',
      status: 'running',
      progress: 0,
      metadata: {
        huggingface_id: 'org/model',
        quantization: 'Q4_K_M',
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading')

    progressStore.removeTask('download_gguf_bundle_org_model_Q4_K_M_1')
    await flushPromises()

    expect(wrapper.text()).not.toContain('Downloading')
  })

  it('clears pending state when matching download task completes', async () => {
    const wrapper = mountModelSearch()
    await flushPromises()
    const progressStore = useProgressStore()

    const downloadBtn = await expandAndGetDownloadButton(wrapper)
    await downloadBtn.trigger('click')
    await flushPromises()

    progressStore.handleEvent('task_updated', {
      task_id: 'download_gguf_bundle_org_model_Q4_K_M_1',
      type: 'download',
      status: 'completed',
      progress: 100,
      metadata: {
        huggingface_id: 'org/model',
        quantization: 'Q4_K_M',
      },
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('Downloading')
  })

  it('ignores non-download task events for pending reconciliation', async () => {
    const wrapper = mountModelSearch()
    await flushPromises()
    const progressStore = useProgressStore()

    const downloadBtn = await expandAndGetDownloadButton(wrapper)
    await downloadBtn.trigger('click')
    await flushPromises()

    progressStore.handleEvent('task_created', {
      task_id: 'build_1',
      type: 'build',
      status: 'running',
      progress: 0,
    })
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading')

    progressStore.removeTask('build_1')
    await flushPromises()

    expect(wrapper.text()).toContain('Downloading')
  })
})

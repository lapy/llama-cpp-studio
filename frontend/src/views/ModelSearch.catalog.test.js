import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { reactive, ref } from 'vue'
import ModelSearch from './ModelSearch.vue'
import { useProgressStore } from '@/stores/progress'

const downloadGgufBundle = vi.fn()
const downloadSafetensorsBundle = vi.fn()
const installCatalogModel = vi.fn()
const searchCatalog = vi.fn()
const fetchModels = vi.fn()
const fetchSafetensorsModels = vi.fn()
const fetchHuggingfaceTokenStatus = vi.fn()

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
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
  downloadGgufBundle,
  downloadSafetensorsBundle,
  installCatalogModel,
  searchCatalog,
  fetchModels,
  fetchSafetensorsModels,
  fetchHuggingfaceTokenStatus,
})

function catalogHfResult() {
  return {
    id: 'huggingface:org/model:gguf',
    provider: 'huggingface',
    provider_item_id: 'org/model:gguf',
    display_name: 'org/model',
    artifact_format: 'gguf',
    source: { id: 'org/model' },
    tasks: ['text-generation'],
    metadata: {
      pipeline_tag: 'text-generation',
      raw: {
        quantizations: {
          Q4_K_M: {
            quantization: 'Q4_K_M',
            files: [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
          },
        },
        mmproj_files: [],
      },
    },
    install_variants: [
      {
        id: 'Q4_K_M',
        label: 'Q4_K_M',
        method: 'direct',
        installable: true,
        files: ['model-Q4_K_M.gguf'],
        size_bytes: 100,
      },
    ],
  }
}

function mountCatalogSearch() {
  return mount(ModelSearch, {
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: {
          props: ['label', 'loading', 'disabled'],
          template: '<button :data-label="label" :data-loading="loading ? \'1\' : \'0\'" :disabled="disabled" @click="$emit(\'click\')">{{ label }}</button>',
        },
        Tag: {
          props: ['value'],
          template: '<span class="tag">{{ value }}</span>',
        },
        InputText: true,
        Dropdown: true,
        LoadingState: true,
        EmptyState: true,
        ProgressTracker: true,
        Dialog: true,
      },
    },
  })
}

describe('ModelSearch catalog integration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    downloadGgufBundle.mockReset()
    downloadSafetensorsBundle.mockReset()
    installCatalogModel.mockReset()
    searchCatalog.mockReset()
    fetchModels.mockReset()
    fetchSafetensorsModels.mockReset()
    fetchHuggingfaceTokenStatus.mockResolvedValue(undefined)
    downloadGgufBundle.mockResolvedValue({})
    fetchModels.mockResolvedValue(undefined)
    fetchSafetensorsModels.mockResolvedValue(undefined)
    searchCatalog.mockImplementation(async () => {
      const data = {
        items: [catalogHfResult()],
        total: 1,
        page: 1,
        has_more: false,
        provider_status: {},
      }
      modelStore.searchResults = data.items
      modelStore.catalogTotal = data.total
      modelStore.searchHasSearched = true
      modelStore.searchLastQuery = 'org'
      return data
    })

    modelStore.searchFormat = 'gguf'
    modelStore.searchQuery = 'org'
    modelStore.searchResults = []
    modelStore.catalogTotal = 0
    modelStore.searchHasSearched = false
    modelStore.searchLastQuery = ''
  })

  async function mountAndSearch() {
    const wrapper = mountCatalogSearch()
    await flushPromises()
    const searchBtn = wrapper.findAll('button').find((btn) => btn.attributes('data-label') === 'Search')
    expect(searchBtn).toBeTruthy()
    await searchBtn.trigger('click')
    await flushPromises()
    return wrapper
  }

  it('downloads Hugging Face catalog variants instead of calling install', async () => {
    const wrapper = await mountAndSearch()

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    expect(downloadBtn.exists()).toBe(true)

    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      null,
      0,
    )
    expect(installCatalogModel).not.toHaveBeenCalled()
  })

  it('reconciles catalog download pending state using Hugging Face repo ids', async () => {
    const wrapper = await mountAndSearch()
    const progressStore = useProgressStore()

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    downloadGgufBundle.mockImplementation(() => new Promise(() => {}))
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadBtn.attributes('data-loading')).toBe('1')

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

    const refreshedBtn = wrapper.find('button[data-label="Download"]')
    expect(refreshedBtn.attributes('data-loading')).toBe('0')
  })
})

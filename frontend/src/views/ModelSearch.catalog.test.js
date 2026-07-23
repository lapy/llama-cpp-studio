import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { reactive, ref } from 'vue'
import ModelSearch from './ModelSearch.vue'
import { useProgressStore } from '@/stores/progress'

const downloadGgufBundle = vi.fn()
const downloadSafetensorsBundle = vi.fn()
const installCatalogModel = vi.fn()
const updateModelProjector = vi.fn()
const updateModelMtp = vi.fn()
const updateModelDflash = vi.fn()
const searchCatalog = vi.fn()
const fetchModels = vi.fn()
const fetchSafetensorsModels = vi.fn()
const fetchHuggingfaceTokenStatus = vi.fn()
const routerPush = vi.fn()
const routerReplace = vi.fn().mockResolvedValue(undefined)
const routeQuery = ref({})

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush, replace: routerReplace }),
  useRoute: () => ({
    name: 'search',
    get query() {
      return routeQuery.value
    },
  }),
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
  updateModelProjector,
  updateModelMtp,
  updateModelDflash,
  searchCatalog,
  fetchModels,
  fetchSafetensorsModels,
  fetchHuggingfaceTokenStatus,
  clearSearchState: vi.fn(),
})

function catalogAudioResult({
  method = 'direct',
  gated = false,
  family = 'qwen3_asr',
  tasks = ['asr'],
  external_inputs_required = false,
  external_inputs_optional = false,
} = {}) {
  return {
    id: 'audio_cpp:Qwen3-ASR-0.6B',
    provider: 'audio_cpp',
    provider_item_id: 'Qwen3-ASR-0.6B',
    display_name: 'Qwen3 ASR 0.6B',
    artifact_format: 'mixed',
    family,
    tasks,
    gated,
    compatible_engines: gated ? [] : ['audio_cpp'],
    input_modalities: ['audio'],
    output_modalities: ['text'],
    source: { kind: 'huggingface_snapshot', repo_id: 'Qwen/Qwen3-ASR-0.6B' },
    install_variants: [
      {
        id: 'Qwen3-ASR-0.6B',
        label: 'Qwen3 ASR 0.6B',
        method,
        method_label:
          method === 'direct'
            ? 'Direct HF'
            : method === 'composite'
              ? 'Assemble (model manager)'
              : 'Convert (model manager)',
        method_hint: method === 'direct' ? 'Downloads a ready snapshot.' : 'Uses model_manager.py',
        installable: true,
        uses_model_manager: method !== 'direct',
        external_inputs_required,
        external_inputs_optional,
      },
    ],
  }
}

function catalogHfResult({
  withProjector = false,
  withMtp = false,
  withDflash = false,
  variantCount = 1,
  gated = false,
  downloads = 1200,
  likes = 45,
  author = 'org',
} = {}) {
  const quantizations = {}
  const install_variants = []
  for (let i = 0; i < variantCount; i += 1) {
    const id = i === 0 ? 'Q4_K_M' : `Q${i}_K`
    quantizations[id] = {
      quantization: id,
      files: [{ filename: `model-${id}.gguf`, size: 100 + i }],
    }
    install_variants.push({
      id,
      label: id,
      method: 'direct',
      installable: true,
      files: [`model-${id}.gguf`],
      size_bytes: 100 + i,
    })
  }
  return {
    id: 'huggingface:org/model:gguf',
    provider: 'huggingface',
    provider_item_id: 'org/model:gguf',
    display_name: 'org/model',
    artifact_format: 'gguf',
    source: { id: 'org/model' },
    tasks: ['text-generation'],
    features: [
      ...(withProjector ? ['multimodal'] : []),
      ...(withMtp ? ['mtp'] : []),
      ...(withDflash ? ['dflash'] : []),
    ],
    gated,
    compatible_engines: ['llama_cpp', 'ik_llama'],
    metadata: {
      pipeline_tag: 'text-generation',
      downloads,
      likes,
      raw: {
        author,
        downloads,
        likes,
        quantizations,
        mmproj_files: withProjector
          ? [
              { filename: 'mmproj-F16.gguf', size: 50 },
              { filename: 'mmproj-F32.gguf', size: 90 },
            ]
          : [],
        mtp_files: withMtp
          ? [
              { filename: 'MTP/mtp-model-Q4_0.gguf', size: 40, label: 'Q4_0' },
              { filename: 'MTP/mtp-model-Q8_0.gguf', size: 80, label: 'Q8_0' },
            ]
          : [],
        dflash_files: withDflash
          ? [
              { filename: 'laguna-s-2.1-DFlash-BF16.gguf', size: 2200, label: 'BF16' },
            ]
          : [],
      },
    },
    install_variants,
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
        InputText: {
          props: ['modelValue', 'placeholder', 'class'],
          emits: ['update:modelValue'],
          template: `<input
            class="input-stub"
            :class="$attrs.class"
            :value="modelValue"
            :placeholder="placeholder"
            @input="$emit('update:modelValue', $event.target.value)"
          >`,
        },
        Dropdown: {
          props: ['modelValue', 'options', 'optionLabel', 'optionValue', 'disabled', 'id'],
          template: `<select
            class="dropdown-stub"
            :id="id"
            :value="modelValue"
            :disabled="disabled"
            :data-option-count="(options || []).length"
            @change="$emit('update:modelValue', $event.target.value)"
          >
            <option
              v-for="opt in options || []"
              :key="String(opt[optionValue || 'value'])"
              :value="opt[optionValue || 'value']"
            >{{ opt[optionLabel || 'label'] }}</option>
          </select>`,
        },
        LoadingState: true,
        EmptyState: true,
        Dialog: {
          props: ['visible', 'header', 'modal', 'style', 'class'],
          emits: ['update:visible'],
          template: `
            <div v-if="visible" class="dialog-stub" :data-header="header">
              <slot />
              <div class="dialog-footer"><slot name="footer" /></div>
            </div>
          `,
        },
      },
    },
  })
}

describe('ModelSearch catalog integration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    routeQuery.value = {}
    routerPush.mockReset()
    routerReplace.mockReset()
    routerReplace.mockResolvedValue(undefined)
    downloadGgufBundle.mockReset()
    downloadSafetensorsBundle.mockReset()
    installCatalogModel.mockReset()
    updateModelProjector.mockReset()
    updateModelMtp.mockReset()
    updateModelDflash.mockReset()
    searchCatalog.mockReset()
    fetchModels.mockReset()
    fetchSafetensorsModels.mockReset()
    fetchHuggingfaceTokenStatus.mockResolvedValue(undefined)
    downloadGgufBundle.mockResolvedValue({})
    updateModelProjector.mockResolvedValue({ applied: true, message: 'Projector applied' })
    updateModelMtp.mockResolvedValue({ applied: true, message: 'MTP draft applied' })
    updateModelDflash.mockResolvedValue({ applied: true, message: 'DFlash draft applied' })
    fetchModels.mockResolvedValue(undefined)
    fetchSafetensorsModels.mockResolvedValue(undefined)
    modelStore.hasHuggingfaceToken = true
    modelStore.allQuantizations = []
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

  async function mountAndSearch(resultFactory = catalogHfResult) {
    searchCatalog.mockImplementation(async () => {
      const data = {
        items: [resultFactory()],
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
      null,
      0,
      null,
      0,
    )
    expect(installCatalogModel).not.toHaveBeenCalled()
  })

  it('shows projector selector and downloads the selected mmproj for vision GGUF models', async () => {
    const wrapper = await mountAndSearch(() => catalogHfResult({ withProjector: true }))

    const projector = wrapper.find('.install-variant__projector select')
    expect(projector.exists()).toBe(true)
    expect(projector.attributes('data-option-count')).toBe('3')
    expect(projector.element.value).toBe('mmproj-F16.gguf')

    await projector.setValue('mmproj-F32.gguf')
    await flushPromises()

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      'mmproj-F32.gguf',
      90,
      null,
      0,
      null,
      0,
    )
  })

  it('hides projector selector when the repo has no mmproj files', async () => {
    const wrapper = await mountAndSearch()
    expect(wrapper.find('.install-variant__projector').exists()).toBe(false)
  })

  it('shows MTP draft selector and downloads the default Q8_0 companion', async () => {
    const wrapper = await mountAndSearch(() => catalogHfResult({ withMtp: true }))

    const mtpSelect = wrapper.findAll('select').find((el) => (el.attributes('id') || '').startsWith('catalog-mtp-'))
    expect(mtpSelect).toBeTruthy()
    expect(mtpSelect.attributes('data-option-count')).toBe('3')
    expect(mtpSelect.element.value).toBe('MTP/mtp-model-Q8_0.gguf')

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      null,
      0,
      'MTP/mtp-model-Q8_0.gguf',
      80,
      null,
      0,
    )
  })

  it('downloads the selected MTP draft when changed from the default', async () => {
    const wrapper = await mountAndSearch(() => catalogHfResult({ withMtp: true }))

    const mtpSelect = wrapper.findAll('select').find((el) => (el.attributes('id') || '').startsWith('catalog-mtp-'))
    await mtpSelect.setValue('MTP/mtp-model-Q4_0.gguf')
    await flushPromises()

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      null,
      0,
      'MTP/mtp-model-Q4_0.gguf',
      40,
      null,
      0,
    )
  })

  it('shows DFlash draft selector and downloads the BF16 companion', async () => {
    const wrapper = await mountAndSearch(() => catalogHfResult({ withDflash: true }))

    expect(wrapper.text()).toContain('DFlash')
    const dflashSelect = wrapper.findAll('select').find((el) => (el.attributes('id') || '').startsWith('catalog-dflash-'))
    expect(dflashSelect).toBeTruthy()
    expect(dflashSelect.element.value).toBe('laguna-s-2.1-DFlash-BF16.gguf')

    const downloadBtn = wrapper.find('button[data-label="Download"]')
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      null,
      0,
      null,
      0,
      'laguna-s-2.1-DFlash-BF16.gguf',
      2200,
    )
  })

  it('opens multi-quant variants in a modal instead of listing them on the card', async () => {
    const wrapper = await mountAndSearch(() => catalogHfResult({ variantCount: 3 }))

    expect(wrapper.find('.install-variant--summary').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Download"]').exists()).toBe(false)

    const browseBtn = wrapper.find('button[data-label="Browse quants"]')
    expect(browseBtn.exists()).toBe(true)
    await browseBtn.trigger('click')
    await flushPromises()

    const dialog = wrapper.find('.dialog-stub')
    expect(dialog.exists()).toBe(true)
    expect(dialog.attributes('data-header')).toContain('Quantizations')
    expect(dialog.findAll('.install-variant')).toHaveLength(3)

    const downloadBtn = dialog.find('button[data-label="Download"]')
    await downloadBtn.trigger('click')
    await flushPromises()

    expect(downloadGgufBundle).toHaveBeenCalledWith(
      'org/model',
      'Q4_K_M',
      [{ filename: 'model-Q4_K_M.gguf', size: 100 }],
      'text-generation',
      null,
      0,
      null,
      0,
      null,
      0,
    )
  })

  it('shows Configure for already downloaded variants', async () => {
    modelStore.allQuantizations = [{
      id: 'org--model--Q4_K_M',
      huggingface_id: 'org/model',
      quantization: 'Q4_K_M',
      filename: 'model-Q4_K_M.gguf',
    }]

    const wrapper = await mountAndSearch()
    const configureBtn = wrapper.find('button[data-label="Configure"]')
    expect(configureBtn.exists()).toBe(true)
    expect(wrapper.find('button[data-label="Download"]').exists()).toBe(false)

    await configureBtn.trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/models/org--model--Q4_K_M/config')
  })

  it('lets downloaded variants reconfigure projector and MTP companions', async () => {
    modelStore.allQuantizations = [{
      id: 'org--model--Q4_K_M',
      huggingface_id: 'org/model',
      quantization: 'Q4_K_M',
      filename: 'model-Q4_K_M.gguf',
      mmproj_filename: 'mmproj-F16.gguf',
      mtp_filename: 'MTP/mtp-model-Q8_0.gguf',
    }]

    const wrapper = await mountAndSearch(() => catalogHfResult({ withProjector: true, withMtp: true }))

    const projector = wrapper.findAll('select').find((el) => (el.attributes('id') || '').startsWith('catalog-projector-'))
    const mtpSelect = wrapper.findAll('select').find((el) => (el.attributes('id') || '').startsWith('catalog-mtp-'))
    expect(projector).toBeTruthy()
    expect(mtpSelect).toBeTruthy()
    expect(projector.element.value).toBe('mmproj-F16.gguf')
    expect(mtpSelect.element.value).toBe('MTP/mtp-model-Q8_0.gguf')
    expect(wrapper.find('button[data-label="Apply projector"]').exists()).toBe(false)
    expect(wrapper.find('button[data-label="Apply MTP"]').exists()).toBe(false)

    await projector.setValue('mmproj-F32.gguf')
    await flushPromises()
    const applyProjector = wrapper.find('button[data-label="Apply projector"]')
    expect(applyProjector.exists()).toBe(true)
    await applyProjector.trigger('click')
    await flushPromises()
    expect(updateModelProjector).toHaveBeenCalledWith('org--model--Q4_K_M', 'mmproj-F32.gguf', 90)

    await mtpSelect.setValue('MTP/mtp-model-Q4_0.gguf')
    await flushPromises()
    const applyMtp = wrapper.find('button[data-label="Apply MTP"]')
    expect(applyMtp.exists()).toBe(true)
    await applyMtp.trigger('click')
    await flushPromises()
    expect(updateModelMtp).toHaveBeenCalledWith('org--model--Q4_K_M', 'MTP/mtp-model-Q4_0.gguf', 40)
  })

  it('shows scan metrics and gated CTA when token is missing', async () => {
    modelStore.hasHuggingfaceToken = false
    const wrapper = await mountAndSearch(() => catalogHfResult({ gated: true }))

    const meta = wrapper.find('.catalog-card__meta')
    expect(meta.exists()).toBe(true)
    expect(meta.text()).toContain('1.2K')
    expect(meta.text()).toContain('45')
    expect(wrapper.find('.catalog-gated-cta').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Set token"]').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('single_file')
  })

  it('keeps previous results visible while a follow-up search is loading', async () => {
    const wrapper = await mountAndSearch()
    expect(wrapper.find('.catalog-card').exists()).toBe(true)

    let resolveSearch
    searchCatalog.mockImplementation(() => {
      modelStore.searchLoading = true
      return new Promise((resolve) => {
        resolveSearch = (data) => {
          modelStore.searchResults = data.items
          modelStore.catalogTotal = data.total
          modelStore.searchLoading = false
          resolve(data)
        }
      })
    })

    const searchBtn = wrapper.findAll('button').find((btn) => btn.attributes('data-label') === 'Search')
    await searchBtn.trigger('click')
    await flushPromises()

    expect(wrapper.find('.catalog-card').exists()).toBe(true)
    expect(wrapper.find('.catalog-results--loading').exists()).toBe(true)
    expect(wrapper.find('.catalog-results__status').exists()).toBe(true)

    resolveSearch({
      items: [catalogHfResult()],
      total: 1,
      page: 1,
      has_more: false,
      provider_status: {},
    })
    await flushPromises()
    expect(wrapper.find('.catalog-results--loading').exists()).toBe(false)
  })

  it('opens audio.cpp install dialog with family override for direct packages', async () => {
    const wrapper = await mountAndSearch(() => catalogAudioResult({ method: 'direct' }))

    expect(wrapper.text()).toContain('result')
    expect(wrapper.text()).not.toContain('verified result')
    expect(wrapper.text()).toContain('qwen3_asr')
    expect(wrapper.text()).toContain('asr')

    const installBtn = wrapper.find('button[data-label="Install"]')
    expect(installBtn.exists()).toBe(true)
    await installBtn.trigger('click')
    await flushPromises()

    const dialog = wrapper.find('.dialog-stub')
    expect(dialog.exists()).toBe(true)
    expect(dialog.attributes('data-header')).toBe('Install audio.cpp package')
    expect(dialog.text()).toContain('Family override')
    expect(dialog.text()).not.toContain('Source file')

    const familyInput = dialog.findAll('input').find((el) => el.attributes('placeholder')?.includes('Detected'))
    expect(familyInput).toBeTruthy()
    expect(familyInput.element.value).toBe('qwen3_asr')

    await dialog.find('button[data-label="Start install"]').trigger('click')
    await flushPromises()

    expect(installCatalogModel).toHaveBeenCalledWith(
      expect.objectContaining({ provider: 'audio_cpp', family: 'qwen3_asr' }),
      expect.objectContaining({ method: 'direct' }),
      expect.objectContaining({ family: 'qwen3_asr' }),
    )
  })

  it('shows converter source inputs for audio.cpp packages that need them', async () => {
    const wrapper = await mountAndSearch(() =>
      catalogAudioResult({
        method: 'converter',
        family: 'citrinet',
        external_inputs_required: true,
      }),
    )

    await wrapper.find('button[data-label="Install"]').trigger('click')
    await flushPromises()

    const dialog = wrapper.find('.dialog-stub')
    expect(dialog.attributes('data-header')).toBe('Prepare audio.cpp package')
    expect(dialog.text()).toContain('Source file')
    expect(dialog.text()).toContain('Source directory')
  })

  it('filters catalog search by install_method facet', async () => {
    modelStore.catalogFacets = {
      install_methods: ['direct', 'composite', 'converter'],
      tasks: ['asr', 'tts'],
    }
    const wrapper = await mountAndSearch(() => catalogAudioResult())

    const methodSelect = wrapper
      .findAll('select.dropdown-stub')
      .find((el) => Array.from(el.element.options || []).some((opt) => opt.value === 'composite'))
    expect(methodSelect).toBeTruthy()
    await methodSelect.setValue('composite')
    await flushPromises()

    const searchBtn = wrapper.findAll('button').find((btn) => btn.attributes('data-label') === 'Search')
    await searchBtn.trigger('click')
    await flushPromises()

    expect(searchCatalog).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        filters: expect.objectContaining({ install_method: 'composite' }),
      }),
    )
  })

  it('shows gated CTA for gated audio.cpp packages without a token', async () => {
    modelStore.hasHuggingfaceToken = false
    const wrapper = await mountAndSearch(() =>
      catalogAudioResult({ gated: true, family: 'pocket_tts', tasks: ['tts'] }),
    )
    expect(wrapper.text()).toContain('Gated')
    expect(wrapper.find('.catalog-gated-cta').exists()).toBe(true)
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

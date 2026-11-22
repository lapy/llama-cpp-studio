import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { reactive } from 'vue'
import EnginesView from './EnginesView.vue'
import { ENGINE_DESCRIPTORS } from '@/engines/registry'
import { engineStoreAccessors } from '@/test-fixtures/engineStoreAccessors'

const toastAdd = vi.fn()
const fetchAll = vi.fn()

vi.mock('vue-router', () => ({
  useRoute: () => ({ hash: '' }),
}))

vi.mock('primevue/useconfirm', () => ({
  useConfirm: () => ({ require: vi.fn() }),
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => ({ subscribe: () => () => {} }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => enginesStore,
}))

const enginesStore = reactive({
  ...engineStoreAccessors(),
  llamaVersions: [],
  ikLlamaVersions: [],
  unslothLlamaVersions: [],
  lmdeployVersions: [],
  onecatVllmVersions: [],
  sglangVersions: [],
  sglangV100Versions: [],
  sglangSxm2Versions: [],
  vllmVersions: [],
  audioCppVersions: [],
  engineDescriptors: [],
  cudaStatus: {},
  systemStatus: {},
  loading: false,
  fetchAll,
  fetchCudaStatus: vi.fn().mockResolvedValue(undefined),
  fetchLmdeployStatus: vi.fn().mockResolvedValue(undefined),
  fetchOnecatVllmStatus: vi.fn().mockResolvedValue(undefined),
  fetchLlamaVersions: vi.fn().mockResolvedValue(undefined),
  fetchSystemStatus: vi.fn().mockResolvedValue(undefined),
  fetchEngineDescriptors: vi.fn().mockResolvedValue([]),
  fetchAudioCppStatus: vi.fn().mockResolvedValue(undefined),
  fetchServerBackedStatuses: vi.fn().mockResolvedValue(undefined),
  checkLlamaCppUpdates: vi.fn().mockResolvedValue(null),
  checkIkLlamaUpdates: vi.fn().mockResolvedValue(null),
  checkUnslothLlamaUpdates: vi.fn().mockResolvedValue({}),
  checkLmdeployUpdates: vi.fn().mockResolvedValue({}),
  checkOnecatVllmUpdates: vi.fn().mockResolvedValue({}),
  checkAudioCppUpdates: vi.fn().mockResolvedValue(null),
  scanEngineParams: vi.fn(),
})

// Descriptors that match no filter: every engine disappears when a filter is
// applied, which is what the page's empty state must explain.
function noMatchDescriptors() {
  return Object.keys(ENGINE_DESCRIPTORS).map((id) => ({
    id,
    label: ENGINE_DESCRIPTORS[id].label || id,
    enabled: true,
    tasks: [],
    artifact_formats: [],
    gpu_archs: [],
    maturity_surfaces: {},
  }))
}

function mountView() {
  return mount(EnginesView, {
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'disabled'],
          emits: ['click'],
          template:
            '<button type="button" :data-label="label" :disabled="disabled" @click="$emit(\'click\')">{{ label }}<slot /></button>',
        },
        Tag: { props: ['value', 'severity'], template: '<span class="tag">{{ value }}</span>' },
        Dialog: {
          props: ['visible'],
          template: `
            <div v-if="visible" class="dialog-stub">
              <header class="dialog-header-stub"><slot name="header" headerId="engine-dialog-title" /></header>
              <slot />
            </div>
          `,
        },
        EngineDialogHeader: {
          props: ['title', 'headingId'],
          template:
            '<div class="eh-stub" :data-title="title" :data-heading-id="headingId"><slot name="leading" /><slot name="tags" /><slot name="actions" /><slot name="more" /></div>',
        },
        EngineCheckUpdatesCta: true,
        EngineBuildSettingsHint: true,
        EngineInstallSourceDialog: true,
        EngineInstallSettingsDialog: true,
        EngineBuildOptionsDialog: true,
        SglangEnginePanel: true,
        GenericEnginePanel: true,
        VersionTable: true,
        InputText: true,
        Select: true,
        ToggleSwitch: true,
        Checkbox: true,
        InputNumber: true,
        ProgressSpinner: true,
      },
    },
  })
}

describe('EnginesView shared page structure', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    fetchAll.mockReset()
    enginesStore.engineDescriptors = []
  })

  it('renders one page header with a count and a single refresh action', async () => {
    fetchAll.mockResolvedValue(undefined)
    const wrapper = mountView()
    await flushPromises()

    const headings = wrapper.findAll('h1')
    expect(headings).toHaveLength(1)
    expect(headings[0].text()).toBe('Engines')
    expect(wrapper.get('.page-header').exists()).toBe(true)
    expect(wrapper.get('.page-header__meta').text()).toContain('0 of')
    const actions = wrapper.findAll('.page-header__actions button')
    expect(actions).toHaveLength(1)
    expect(actions[0].attributes('data-label')).toBe('Refresh')
    wrapper.unmount()
  })

  it('wraps the catalog in a labelled section with an accessible collapse control', async () => {
    fetchAll.mockResolvedValue(undefined)
    const wrapper = mountView()
    await flushPromises()

    const section = wrapper.get('section#runtime-engines')
    const heading = widgetSectionHeading(section)
    expect(heading.text()).toBe('Runtime engines')
    expect(section.attributes('aria-labelledby')).toBe(heading.attributes('id'))

    const toggle = section.findAll('.page-section__actions button')[0]
    expect(toggle.attributes('aria-label')).toBe('Collapse runtime engines')
    expect(toggle.attributes('aria-expanded')).toBe('true')
    expect(toggle.attributes('aria-controls')).toBe('engine-catalog-body')
    expect(wrapper.find('#engine-catalog-body').exists()).toBe(true)

    await toggle.trigger('click')
    expect(toggle.attributes('aria-expanded')).toBe('false')
    expect(toggle.attributes('aria-label')).toBe('Expand runtime engines')
    // The disclosure target stays in the DOM so aria-controls keeps resolving.
    expect(wrapper.get('#engine-catalog-body').element.style.display).toBe('none')
    wrapper.unmount()
  })

  it('shows a loading state until the first catalog load settles', async () => {
    let resolveCatalog
    fetchAll.mockReturnValue(
      new Promise((resolve) => {
        resolveCatalog = resolve
      }),
    )
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.get('.page-loading').text()).toContain('Loading runtime engines')
    expect(wrapper.find('.engine-grid').exists()).toBe(false)

    resolveCatalog(undefined)
    await flushPromises()
    expect(wrapper.find('.page-loading').exists()).toBe(false)
    expect(wrapper.find('.engine-grid').exists()).toBe(true)
    wrapper.unmount()
  })

  it('uses the shared toolbar, summary and reset for filters', async () => {
    fetchAll.mockResolvedValue(undefined)
    const wrapper = mountView()
    await flushPromises()

    const toolbar = wrapper.get('.engine-filters')
    expect(toolbar.classes()).toContain('page-toolbar')
    expect(toolbar.attributes('role')).toBe('group')
    expect(toolbar.attributes('aria-label')).toBe('Engine filters')
    expect(toolbar.get('.page-toolbar__summary').text()).toContain('engines')
    expect(wrapper.get('#engine-task-filter').exists()).toBe(true)
    expect(wrapper.get('#engine-hardware-filter').exists()).toBe(true)
    expect(toolbar.get('.page-toolbar__actions .engine-filters__reset').exists()).toBe(true)
    wrapper.unmount()
  })

  it('explains an over-filtered catalog and resets it from the empty state', async () => {
    fetchAll.mockResolvedValue(undefined)
    enginesStore.engineDescriptors = noMatchDescriptors()
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('#engine-task-filter').setValue('audio')
    await flushPromises()

    const empty = wrapper.get('.page-empty')
    expect(empty.attributes('role')).toBe('status')
    expect(empty.text()).toContain('No engines match these filters')
    expect(empty.text()).toContain('hide every runtime')
    expect(wrapper.find('.engine-grid').exists()).toBe(false)
    expect(wrapper.get('.engine-filters').get('.page-toolbar__summary').text()).toContain('0 of')

    await empty.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.find('.page-empty').exists()).toBe(false)
    expect(wrapper.findAll('.engine-card').length).toBeGreaterThan(0)
    wrapper.unmount()
  })

  it('labels the engine dialog with the heading id PrimeVue generates', async () => {
    fetchAll.mockResolvedValue(undefined)
    const wrapper = mountView()
    await flushPromises()

    wrapper.vm.openEngineModal(Object.keys(ENGINE_DESCRIPTORS)[0])
    await flushPromises()

    const header = wrapper.get('.eh-stub')
    expect(header.attributes('data-title')).toBe('llama.cpp')
    expect(header.attributes('data-heading-id')).toBe('engine-dialog-title')
    const body = wrapper.get('.engine-modal-body')
    expect(body.classes()).toContain('dialog-stack')
    wrapper.unmount()
  })
})

function widgetSectionHeading(section) {
  return section.get('.page-section__title')
}

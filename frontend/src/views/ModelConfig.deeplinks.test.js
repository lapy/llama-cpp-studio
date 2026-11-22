import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, shallowMount } from '@vue/test-utils'
import { reactive } from 'vue'
import { AUDIO_ENGINE_IDS, ENGINE_DESCRIPTORS } from '@/engines/registry'
import {
  MODEL_CONFIG_OVERVIEW_TAB,
  sectionHashForTab,
  tabIdForHash,
} from '@/composables/useModelConfigSections'
import { readDraft, writeDraft } from '@/composables/useDraftGuard'
import ModelConfig from './ModelConfig.vue'

enableAutoUnmount(afterEach)

const ENGINE_ID = Object.keys(ENGINE_DESCRIPTORS)[0]
const route = reactive({
  params: { id: 'model-1' },
  path: '/models/model-1/config',
  query: {},
  hash: '',
})
const routerPush = vi.hoisted(() => vi.fn())

vi.mock('axios', () => ({
  default: {
    defaults: {},
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
  },
}))

vi.mock('vue-router', () => ({
  useRoute: () => route,
  useRouter: () => ({ push: routerPush }),
  onBeforeRouteLeave: () => {},
  RouterLink: { props: ['to'], template: '<a><slot /></a>' },
}))

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: vi.fn() }) }))
vi.mock('primevue/useconfirm', () => ({ useConfirm: () => ({ require: vi.fn() }) }))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    models: [
      {
        base_model_name: 'Test Model',
        huggingface_id: 'org/model',
        quantizations: [
          {
            id: 'model-1',
            display_name: 'Test Model',
            base_model_name: 'Test Model',
            huggingface_id: 'org/model',
            quantization: 'Q4_K_M',
            format: 'gguf',
            artifact: {},
          },
        ],
      },
    ],
    allQuantizations: [],
    fetchModels: vi.fn(),
  }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    engineDescriptors: [],
    runtimePlanState: { applicable: false, stale: false },
    runtimePlan: { applicable: false, pending: false, changes: [], reason: null },
    fetchEngineDescriptors: vi.fn().mockResolvedValue([]),
    fetchRuntimePlan: vi.fn().mockResolvedValue(undefined),
    fetchRuntimePlanState: vi.fn().mockResolvedValue(undefined),
    fetchGpuList: vi.fn().mockResolvedValue({
      vendor: null,
      device_count: 0,
      gpus: [],
      cpu_only_mode: true,
    }),
    invalidateRuntimePlan: vi.fn(),
  }),
}))

import axios from 'axios'

const buttonStub = {
  props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'rounded'],
  emits: ['click'],
  template:
    '<button type="button" :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
}

const configResponse = {
  engine: ENGINE_ID,
  engines: {
    [ENGINE_ID]: {
      temperature: 0.7,
    },
  },
}

function draftPayload(temperature) {
  return {
    engine: ENGINE_ID,
    config: {
      engine: ENGINE_ID,
      engines: { [ENGINE_ID]: { temperature } },
      temperature,
    },
    activeParamKeys: ['temperature'],
  }
}

let originalScrollIntoView
let scrollCalls

beforeEach(() => {
  sessionStorage.clear()
  route.params.id = 'model-1'
  route.path = '/models/model-1/config'
  route.query = {}
  route.hash = ''
  routerPush.mockReset()
  scrollCalls = []
  originalScrollIntoView = HTMLElement.prototype.scrollIntoView
  HTMLElement.prototype.scrollIntoView = function scrollIntoView(options) {
    scrollCalls.push({ id: this.id, options })
  }

  vi.mocked(axios.get).mockReset()
  vi.mocked(axios.post).mockReset()
  vi.mocked(axios.put).mockReset()
  vi.mocked(axios.get).mockImplementation((url) => {
    if (url === '/api/models/model-1/config') return Promise.resolve({ data: configResponse })
    if (url === '/api/models/param-registry') {
      return Promise.resolve({
        data: {
          sections: [
            {
              id: 'sampling',
              label: 'Sampling',
              params: [
                {
                  key: 'temperature',
                  label: 'Temperature',
                  type: 'float',
                  scalar_type: 'float',
                  value_kind: 'scalar',
                  default: 0.8,
                  primary_flag: '--temperature',
                  flags: ['--temperature'],
                  supported: true,
                },
              ],
            },
          ],
          scan_error: null,
          scan_pending: false,
        },
      })
    }
    if (url === '/api/gpu-list') {
      return Promise.resolve({
        data: { vendor: null, device_count: 0, gpus: [], cpu_only_mode: true },
      })
    }
    throw new Error(`Unexpected GET ${url}`)
  })
  vi.mocked(axios.put).mockResolvedValue({ data: configResponse })
})

afterEach(() => {
  if (originalScrollIntoView) HTMLElement.prototype.scrollIntoView = originalScrollIntoView
  else delete HTMLElement.prototype.scrollIntoView
})

async function mountReady() {
  const wrapper = shallowMount(ModelConfig, {
    attachTo: document.body,
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: buttonStub,
        ModelConfigActionBar: false,
        ModelConfigNamesSummary: false,
        PageSection: false,
        PageToolbar: false,
        SectionTabs: false,
        StateBanner: false,
      },
    },
  })
  await vi.waitFor(() => expect(wrapper.vm.loading).toBe(false))
  await flushPromises()
  return wrapper
}

function tabByLabel(wrapper, label) {
  return wrapper.findAll('.section-tabs [role="tab"]').find((tab) => tab.text().includes(label))
}

describe('ModelConfig deep-linkable sections', () => {
  it('maps audio labels and ids to their real tabs', () => {
    const tabs = ENGINE_DESCRIPTORS[AUDIO_ENGINE_IDS[0]].configTabs
    expect(sectionHashForTab(MODEL_CONFIG_OVERVIEW_TAB)).toBe('overview')
    expect(tabIdForHash('#overview', tabs)).toBe('')
    expect(tabIdForHash('#overview', [MODEL_CONFIG_OVERVIEW_TAB, ...tabs])).toBe('overview')
    expect(tabIdForHash('#runtime', tabs)).toBe('server')
    expect(tabIdForHash('#defaults', tabs)).toBe('api')
    expect(tabIdForHash('#api', tabs)).toBe('api')
    expect(sectionHashForTab(tabs[0])).toBe('runtime')
  })

  it('opens the overview from its deep link', async () => {
    route.hash = '#overview'
    const wrapper = await mountReady()

    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('overview'))
    await vi.waitFor(() =>
      expect(scrollCalls).toEqual(
        expect.arrayContaining([expect.objectContaining({ id: 'config-panel-overview' })]),
      ),
    )
  })

  it('opens a section inside a tab from the initial hash and follows hashchange', async () => {
    route.hash = '#launch'
    const wrapper = await mountReady()

    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('launch'))
    await vi.waitFor(() =>
      expect(scrollCalls).toEqual(
        expect.arrayContaining([expect.objectContaining({ id: 'config-panel-launch' })]),
      ),
    )

    const focused = document.createElement('input')
    document.body.appendChild(focused)
    focused.focus()
    route.hash = '#runtime'
    await flushPromises()
    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('runtime'))
    await vi.waitFor(() =>
      expect(scrollCalls).toEqual(
        expect.arrayContaining([expect.objectContaining({ id: 'config-panel-runtime' })]),
      ),
    )
    expect(document.activeElement).toBe(focused)
    focused.remove()
  })

  it('falls back to the default section when the hash is cleared', async () => {
    route.hash = '#launch'
    const wrapper = await mountReady()
    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('launch'))

    route.hash = ''
    await flushPromises()

    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('overview'))
  })

  it('keeps the overview default without a hash and puts a clicked tab in the URL', async () => {
    const wrapper = await mountReady()

    expect(wrapper.vm.pageTab).toBe('overview')
    expect(scrollCalls).toHaveLength(0)

    await tabByLabel(wrapper, 'Launch').trigger('click')
    await flushPromises()

    expect(wrapper.vm.pageTab).toBe('launch')
    expect(routerPush).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/models/model-1/config', hash: '#launch' }),
    )
  })

  it('navigates the task sections through the shared tab strip with matching panel ids', async () => {
    const wrapper = await mountReady()

    const tabs = wrapper.findAll('.section-tabs [role="tab"]')
    expect(tabs.map((tab) => tab.text())).toEqual(['Overview', 'Runtime', 'Names', 'Launch'])

    const namesTab = wrapper.get('#config-tab-names')
    expect(namesTab.attributes('aria-controls')).toBe('config-panel-names')
    expect(namesTab.attributes('aria-selected')).toBe('false')
    const namesPanel = wrapper.get('#config-panel-names')
    expect(namesPanel.attributes('aria-labelledby')).toBe('config-tab-names')
    expect(namesPanel.text()).toContain('Names & routing')
    expect(namesPanel.text()).toContain('Catalog identity')

    await namesTab.trigger('click')
    await flushPromises()

    expect(wrapper.vm.pageTab).toBe('names')
    expect(namesTab.attributes('aria-selected')).toBe('true')
    expect(wrapper.get('#config-tab-names').attributes('tabindex')).toBe('0')
    expect(routerPush).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/models/model-1/config', hash: '#names' }),
    )
  })

  it('opens the names section from its deep link', async () => {
    route.hash = '#names'
    const wrapper = await mountReady()

    await vi.waitFor(() => expect(wrapper.vm.pageTab).toBe('names'))
    await vi.waitFor(() =>
      expect(scrollCalls).toEqual(
        expect.arrayContaining([expect.objectContaining({ id: 'config-panel-names' })]),
      ),
    )
  })
})

describe('ModelConfig draft survival', () => {
  it('restores a draft with a visible notice and clears it after a successful save', async () => {
    writeDraft('model-1', ENGINE_ID, draftPayload(0.2))
    const wrapper = await mountReady()

    await vi.waitFor(() => expect(wrapper.text()).toContain('Restore draft'))
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Restore draft')
      .trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Restored unsaved changes')
    expect(wrapper.vm.config.temperature).toBe(0.2)
    expect(readDraft('model-1', ENGINE_ID)).not.toBeNull()

    await wrapper.get('button[data-label="Save changes"]').trigger('click')
    await flushPromises()

    expect(readDraft('model-1', ENGINE_ID)).toBeNull()
    expect(wrapper.text()).not.toContain('Restored unsaved changes')
  })

  it('clears the draft on an explicit discard', async () => {
    writeDraft('model-1', ENGINE_ID, draftPayload(0.2))
    const wrapper = await mountReady()

    await vi.waitFor(() => expect(wrapper.text()).toContain('Discard draft'))
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Discard draft')
      .trigger('click')
    await flushPromises()

    expect(readDraft('model-1', ENGINE_ID)).toBeNull()
    expect(wrapper.text()).not.toContain('Restore draft')
  })

  it('does not offer another model draft', async () => {
    writeDraft('model-2', ENGINE_ID, draftPayload(0.2))
    const wrapper = await mountReady()

    expect(wrapper.text()).not.toContain('Restore draft')
    expect(wrapper.text()).not.toContain('Restored unsaved changes')
    expect(readDraft('model-2', ENGINE_ID)).not.toBeNull()
  })
})

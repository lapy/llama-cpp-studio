import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
  RouterLink: {
    name: 'RouterLink',
    props: ['to'],
    template: '<a :href="to"><slot /></a>',
  },
}))

const fetchEngineDescriptors = vi.fn()

const enginesStore = reactive({
  engineDescriptors: [],
  systemStatus: { proxy_status: { healthy: true } },
  fetchEngineDescriptors,
})

const modelStore = reactive({
  models: [],
  startModel: vi.fn(),
  fetchModels: vi.fn(),
})

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => enginesStore,
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => modelStore,
}))

import SetupChecklist from './SetupChecklist.vue'

const DISMISS_KEY = 'llama-studio.setup-checklist.dismissed'

const READY_ENGINE = [
  { id: 'engine-a', label: 'Engine A', runnable: true, installed_versions: 1, enabled: true },
]
const INACTIVE_ENGINE = [
  { id: 'engine-a', label: 'Engine A', runnable: false, installed_versions: 2, enabled: true },
  { id: 'engine-b', label: 'Engine B', runnable: false, installed_versions: 0, enabled: true },
]

function modelsWith({ reviewed = false, active = false, runtimeQuality = 'verified' } = {}) {
  return [
    {
      quantizations: [
        {
          id: 'm1',
          is_active: active,
          config_reviewed: reviewed,
          runtime_quality: runtimeQuality,
        },
      ],
    },
  ]
}

function mountChecklist() {
  return mount(SetupChecklist)
}

function currentLabel(wrapper) {
  return wrapper.get('.setup-checklist h2').text()
}

function currentAction(wrapper) {
  return wrapper.get('.setup-checklist__link').text()
}

async function mountReady() {
  const wrapper = mountChecklist()
  await flushPromises()
  return wrapper
}

beforeEach(() => {
  localStorage.clear()
  routerPush.mockReset()
  fetchEngineDescriptors.mockReset().mockResolvedValue([])
  modelStore.models = []
  modelStore.startModel.mockReset().mockResolvedValue(undefined)
  modelStore.fetchModels.mockReset().mockResolvedValue(undefined)
  enginesStore.engineDescriptors = []
  enginesStore.systemStatus = { proxy_status: { healthy: true } }
})

describe('SetupChecklist', () => {
  it('starts with finding a model and links to Search', async () => {
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Find a model')
    expect(wrapper.text()).toContain('Search for a GGUF model')
    expect(currentAction(wrapper)).toBe('Search models')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/search')
    expect(wrapper.text()).toContain('Show all steps')
    expect(wrapper.text()).toContain('0 of 5 steps complete')
    wrapper.unmount()
  })

  it('asks to install an engine once a model is in the library', async () => {
    modelStore.models = modelsWith()
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Install an engine')
    expect(wrapper.text()).toContain('Install and activate an engine')
    expect(currentAction(wrapper)).toBe('Install an engine')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/engines')
    wrapper.unmount()
  })

  it('asks to activate an installed but inactive engine instead of reinstalling', async () => {
    modelStore.models = modelsWith()
    enginesStore.engineDescriptors = INACTIVE_ENGINE
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Activate an engine')
    expect(currentAction(wrapper)).toBe('Activate an engine')
    expect(wrapper.text()).toContain('Installed but not active: Engine A')
    expect(wrapper.text()).not.toContain('Install and activate')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/engines')
    wrapper.unmount()
  })

  it('marks the model and engine done and moves to review configuration', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith()
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Review configuration')
    expect(currentAction(wrapper)).toBe('Configure model')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/models/m1/config')

    await wrapper.get('.setup-checklist__expand').trigger('click')
    const steps = wrapper.findAll('.setup-checklist__steps li')
    expect(steps).toHaveLength(5)
    expect(steps[0].text()).toContain('Model in library')
    expect(steps[0].classes()).toContain('is-done')
    expect(steps[1].text()).toContain('Engine ready')
    expect(steps[1].classes()).toContain('is-done')
    expect(steps[2].classes()).toContain('is-current')
    expect(steps[2].text()).toContain('→')
    expect(wrapper.text()).toContain('2 of 5 steps complete')
    wrapper.unmount()
  })

  it('renders every step as a deep link to its setup surface', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith()
    const wrapper = await mountReady()

    await wrapper.get('.setup-checklist__expand').trigger('click')
    const hrefs = wrapper
      .findAll('.setup-checklist__step-link')
      .map((link) => link.attributes('href'))
    expect(hrefs).toEqual(['/search', '/engines', '/models/m1/config', '/proxy'])
    wrapper.unmount()
  })

  it('does not mark configuration reviewed just because a model was downloaded', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith()
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Review configuration')
    expect(currentAction(wrapper)).toBe('Configure model')
    wrapper.unmount()
  })

  it('offers an explicit Open proxy action when a running model cannot be verified', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    enginesStore.systemStatus = { proxy_status: { healthy: false } }
    modelStore.models = modelsWith({ reviewed: true, active: true, runtimeQuality: 'stale' })
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Connect a client')
    expect(currentAction(wrapper)).toBe('Open proxy')
    expect(wrapper.text()).toContain('inference proxy is offline')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/proxy')
    wrapper.unmount()
  })

  it('routes to the proxy when proxy status is unknown but a model is running', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    enginesStore.systemStatus = {}
    modelStore.models = modelsWith({ reviewed: true, active: true })
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Connect a client')
    expect(currentAction(wrapper)).toBe('Open proxy')
    expect(wrapper.text()).toContain('Proxy status is unavailable')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/proxy')
    wrapper.unmount()
  })

  it('starts the configured model and advances when the proxy verifies it', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith({ reviewed: true })
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Start the model')
    expect(currentAction(wrapper)).toBe('Start configured model')
    await wrapper.get('.setup-checklist__link').trigger('click')
    await flushPromises()
    expect(modelStore.startModel).toHaveBeenCalledWith('m1')
    expect(modelStore.fetchModels).toHaveBeenCalled()

    modelStore.models = modelsWith({ reviewed: true, active: true })
    await flushPromises()
    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    wrapper.unmount()
  })

  it('stays hidden after success and returns only when the state regresses', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith({ reviewed: true, active: true })
    const wrapper = await mountReady()

    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    expect(wrapper.find('.setup-reopen').exists()).toBe(false)

    modelStore.models = modelsWith({ reviewed: true })
    await flushPromises()
    expect(currentLabel(wrapper)).toBe('Start the model')
    wrapper.unmount()
  })

  it('hides the guide while a model is loading and returns if it stops', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = [
      {
        quantizations: [
          {
            id: 'm1',
            is_active: false,
            config_reviewed: true,
            status: 'loading',
            runtime_quality: 'verified',
          },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(wrapper.find('.setup-checklist').exists()).toBe(false)

    modelStore.models = modelsWith({ reviewed: true })
    await flushPromises()
    expect(currentLabel(wrapper)).toBe('Start the model')
    wrapper.unmount()
  })

  it('keeps dismissal and reopens the guide from the persistent control', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = modelsWith()
    localStorage.setItem(DISMISS_KEY, '1')
    const wrapper = await mountReady()

    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    const reopen = wrapper.get('.setup-reopen')
    expect(reopen.text()).toBe('Show setup guide')
    await reopen.trigger('click')
    await flushPromises()

    expect(wrapper.find('.setup-checklist').exists()).toBe(true)
    expect(localStorage.getItem(DISMISS_KEY)).toBeNull()
    expect(wrapper.findAll('.setup-checklist__steps li')).toHaveLength(5)
    wrapper.unmount()
  })

  it('reports an unknown engine state when descriptors cannot be loaded', async () => {
    modelStore.models = modelsWith()
    fetchEngineDescriptors.mockRejectedValueOnce(new Error('offline'))
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Check your engine')
    expect(currentAction(wrapper)).toBe('Open engines')
    expect(wrapper.text()).not.toContain('Install and activate')
    expect(wrapper.get('.setup-checklist__link').attributes('href')).toBe('/engines')
    wrapper.unmount()
  })
})

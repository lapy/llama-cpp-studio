import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPush }),
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
  it('asks for an install when no engine is installed', async () => {
    enginesStore.engineDescriptors = []
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Install an engine')
    expect(wrapper.text()).toContain('Install and activate an engine')
    expect(currentAction(wrapper)).toBe('Install an engine')
    expect(wrapper.text()).toContain('Show all steps')

    await wrapper.get('.setup-checklist__link').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/engines')
    wrapper.unmount()
  })

  it('asks to activate an installed but inactive engine instead of reinstalling', async () => {
    enginesStore.engineDescriptors = INACTIVE_ENGINE
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Activate an engine')
    expect(currentAction(wrapper)).toBe('Activate an engine')
    expect(wrapper.text()).toContain('Installed but not active: Engine A')
    expect(wrapper.text()).not.toContain('Install and activate')

    await wrapper.get('.setup-checklist__link').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/engines')
    wrapper.unmount()
  })

  it('marks a runnable engine done and moves to the first model download', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Download a model')
    expect(currentAction(wrapper)).toBe('Search and download')
    await wrapper.get('.setup-checklist__link').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/search')

    await wrapper.get('.setup-checklist__expand').trigger('click')
    const steps = wrapper.findAll('.setup-checklist__steps li')
    expect(steps).toHaveLength(5)
    expect(steps[0].text()).toContain('Engine ready')
    expect(steps[0].classes()).toContain('is-done')
    expect(steps[1].classes()).toContain('is-current')
    expect(steps[1].text()).toContain('→')
    wrapper.unmount()
  })

  it('does not mark configuration reviewed just because a model was downloaded', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: false, config_reviewed: false, runtime_quality: 'verified' },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Review configuration')
    expect(currentAction(wrapper)).toBe('Configure model')
    await wrapper.get('.setup-checklist__link').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/models/m1/config')
    wrapper.unmount()
  })

  it('offers an explicit Open proxy action when a running model cannot be verified', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    enginesStore.systemStatus = { proxy_status: { healthy: false } }
    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: true, config_reviewed: true, runtime_quality: 'stale' },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Ready to connect')
    expect(currentAction(wrapper)).toBe('Open proxy')
    expect(wrapper.text()).toContain('inference proxy is offline')
    await wrapper.get('.setup-checklist__link').trigger('click')
    expect(routerPush).toHaveBeenCalledWith('/proxy')
    wrapper.unmount()
  })

  it('routes to the proxy when proxy status is unknown but a model is running', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    enginesStore.systemStatus = {}
    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: true, config_reviewed: true, runtime_quality: 'verified' },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Ready to connect')
    expect(currentAction(wrapper)).toBe('Open proxy')
    expect(wrapper.text()).toContain('Proxy status is unavailable')
    wrapper.unmount()
  })

  it('starts the configured model and advances when the proxy verifies it', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: false, config_reviewed: true, runtime_quality: 'verified' },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Start the model')
    expect(currentAction(wrapper)).toBe('Start configured model')
    await wrapper.get('.setup-checklist__link').trigger('click')
    await flushPromises()
    expect(modelStore.startModel).toHaveBeenCalledWith('m1')
    expect(modelStore.fetchModels).toHaveBeenCalled()

    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: true, config_reviewed: true, runtime_quality: 'verified' },
        ],
      },
    ]
    await flushPromises()
    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    wrapper.unmount()
  })

  it('stays hidden after success and returns only when the state regresses', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: true, config_reviewed: true, runtime_quality: 'verified' },
        ],
      },
    ]
    const wrapper = await mountReady()

    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    expect(wrapper.find('.setup-reopen').exists()).toBe(false)

    modelStore.models = [
      {
        quantizations: [
          { id: 'm1', is_active: false, config_reviewed: true, runtime_quality: 'verified' },
        ],
      },
    ]
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

    modelStore.models = [
      {
        quantizations: [
          {
            id: 'm1',
            is_active: false,
            config_reviewed: true,
            status: 'stopped',
            runtime_quality: 'verified',
          },
        ],
      },
    ]
    await flushPromises()
    expect(currentLabel(wrapper)).toBe('Start the model')
    wrapper.unmount()
  })

  it('keeps dismissal and reopens the guide from the persistent control', async () => {
    enginesStore.engineDescriptors = READY_ENGINE
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
    fetchEngineDescriptors.mockRejectedValueOnce(new Error('offline'))
    const wrapper = await mountReady()

    expect(currentLabel(wrapper)).toBe('Check your engine')
    expect(currentAction(wrapper)).toBe('Open engines')
    expect(wrapper.text()).not.toContain('Install and activate')
    wrapper.unmount()
  })
})

import { describe, expect, it, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive } from 'vue'

const fetchEngineDescriptors = vi.fn().mockResolvedValue([])
const enginesStore = reactive({
  engineDescriptors: [],
  systemStatus: { proxy_status: { healthy: true } },
  fetchEngineDescriptors,
})
const modelStore = reactive({ models: [] })

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => enginesStore,
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => modelStore,
}))

import SetupChecklist from './SetupChecklist.vue'

function mountChecklist() {
  return mount(SetupChecklist)
}

describe('SetupChecklist', () => {
  it('does not treat an inactive install as ready and waits for descriptors', async () => {
    enginesStore.engineDescriptors = [
      { id: 'llama_cpp', label: 'llama.cpp', runnable: false, installed_versions: 1, enabled: true },
      { id: 'unsloth_llama', label: 'Unsloth llama.cpp', runnable: false, installed_versions: 0, enabled: true },
    ]
    modelStore.models = []
    const wrapper = mountChecklist()
    await flushPromises()
    expect(wrapper.text()).toContain('Prepare a runnable engine')
    expect(wrapper.text()).toContain('Unsloth')
    expect(wrapper.text()).toContain('Show all steps')
    expect(wrapper.findAll('.setup-checklist__steps li')).toHaveLength(0)
    await wrapper.get('.setup-checklist__expand').trigger('click')
    expect(wrapper.findAll('.setup-checklist__steps li').length).toBeGreaterThan(1)
    wrapper.unmount()
  })

  it('does not mark configuration reviewed just because a model was downloaded', async () => {
    enginesStore.engineDescriptors = [
      { id: 'unsloth_llama', label: 'Unsloth llama.cpp', runnable: true, enabled: true },
    ]
    modelStore.models = [{
      quantizations: [{ id: 'm1', is_active: false, config_reviewed: false, runtime_quality: 'verified' }],
    }]
    const wrapper = mountChecklist()
    await flushPromises()
    expect(wrapper.text()).toContain('Review configuration')
    wrapper.unmount()
  })

  it('hides onboarding after a verified running model', async () => {
    enginesStore.engineDescriptors = [
      { id: 'llama_cpp', label: 'llama.cpp', runnable: true, enabled: true },
    ]
    enginesStore.systemStatus = { proxy_status: { healthy: true } }
    modelStore.models = [{
      quantizations: [{ id: 'm1', is_active: true, config_reviewed: true, runtime_quality: 'verified' }],
    }]
    const wrapper = mountChecklist()
    await flushPromises()
    expect(wrapper.find('.setup-checklist').exists()).toBe(false)
    wrapper.unmount()
  })
})

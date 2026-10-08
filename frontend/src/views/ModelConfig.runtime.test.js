import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, shallowMount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import axios from 'axios'
import ModelConfig from './ModelConfig.vue'
import { useEnginesStore } from '@/stores/engines'

enableAutoUnmount(afterEach)
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
  useRoute: () => ({ params: { id: 'model-1' } }),
  useRouter: () => ({ push: vi.fn() }),
  onBeforeRouteLeave: vi.fn(),
}))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: vi.fn() }) }))

let pending
let applied
let running
let pinia
beforeEach(() => {
  pinia = createPinia()
  setActivePinia(pinia)
  applied = false
  running = true
  pending = { applicable: true, pending: true, changes: ['Saved settings changed'] }
  sessionStorage.clear()
  vi.mocked(axios.get).mockReset()
  vi.mocked(axios.post).mockReset()
  vi.mocked(axios.get).mockImplementation(async (url) => {
    if (url === '/api/models') {
      return { data: [{ base_model_name: 'Example', quantizations: [{
        id: 'model-1', format: 'gguf', is_active: running, runtime_quality: 'verified',
      }] }] }
    }
    if (url === '/api/models/model-1/config') {
      return { data: { engine: 'llama_cpp', engines: { llama_cpp: {} } } }
    }
    if (url === '/api/models/param-registry') return { data: { sections: [] } }
    if (url === '/api/gpu-list') return { data: { gpus: [] } }
    if (url === '/api/engines') return { data: { engines: [] } }
    if (url === '/api/llama-swap/pending') return { data: pending }
    if (url === '/api/llama-swap/stale') return { data: { applicable: true, stale: !applied } }
    throw new Error(`Unexpected GET ${url}`)
  })
  vi.mocked(axios.post).mockImplementation(async (url) => {
    if (url !== '/api/models/model-1/runtime/apply') throw new Error(`Unexpected POST ${url}`)
    applied = true
    pending = { ...pending, pending: false, models: [] }
    return { data: { status: 'succeeded' } }
  })
})

async function mountConfig() {
  useEnginesStore().markSwapConfigStaleLocal()
  const wrapper = shallowMount(ModelConfig, {
    global: {
      plugins: [pinia],
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label'], emits: ['click'],
          template: '<button :data-label="label" @click="$emit(\'click\')">{{ label }}</button>',
        },
        Dialog: {
          props: ['visible'],
          template: '<div v-if="visible" class="dialog"><slot /><slot name="footer" /></div>',
        },
      },
    },
  })
  await vi.waitFor(() => expect(wrapper.vm.loading).toBe(false))
  await flushPromises()
  return wrapper
}

describe('model configuration with real runtime stores', () => {
  it('keeps legacy reload available after the store normalizes a missing models array', async () => {
    const wrapper = await mountConfig()
    expect(useEnginesStore().swapConfigPending.models).toEqual([])
    expect(useEnginesStore().swapConfigPending.launch_manifests).toBe(false)
    expect(wrapper.find('button[data-label="Reload proxy"]').exists()).toBe(true)
    expect(wrapper.get('.runtime-state__detail').text()).toContain('Pending changes')
  })

  it.each([
    ['restart_now', true, 'Restart this model'],
    ['publish_next_start', false, 'Use on next start'],
  ])('clears the %s plan after successful apply', async (action, isRunning, label) => {
    running = isRunning
    pending = {
      ...pending, launch_manifests: true, requires_proxy_reload: false, plan_id: 'plan',
      models: [{ catalog_id: 'model-1', model_id: 'model-1', action, running,
        desired_revision: 'new', published_revision: 'old', requires_proxy_reload: false }],
    }
    const wrapper = await mountConfig()
    await wrapper.get(`button[data-label="${label}"]`).trigger('click')
    await flushPromises()
    await wrapper.get(`.dialog button[data-label="${label}"]`).trigger('click')
    await vi.waitFor(() => expect(wrapper.vm.applyingLlamaSwap).toBe(false))
    await flushPromises()

    expect(applied).toBe(true)
    expect(useEnginesStore().swapConfigPending.models).toEqual([])
    expect(wrapper.find(`button[data-label="${label}"]`).exists()).toBe(false)
    expect(wrapper.get('.runtime-state__detail').text()).toContain(
      running ? 'this model is running them' : 'The model is stopped',
    )
  })

  it('follows a running apply and can stop it before it finishes', async () => {
    running = true
    pending = {
      ...pending, launch_manifests: true, requires_proxy_reload: false, plan_id: 'plan',
      models: [{ catalog_id: 'model-1', model_id: 'model-1', action: 'restart_now', running: true,
        desired_revision: 'new', published_revision: 'old', requires_proxy_reload: false }],
    }
    const previousGet = vi.mocked(axios.get).getMockImplementation()
    let polls = 0
    vi.mocked(axios.post).mockImplementation(async (url) => {
      if (url.endsWith('/cancel')) return { data: { status: 'running', cancel_requested: true } }
      return { data: { status: 'running', operation_id: 'op1', message: 'Apply accepted' } }
    })
    vi.mocked(axios.get).mockImplementation(async (url) => {
      if (url === '/api/models/model-1/runtime/apply/op1') {
        polls += 1
        return {
          data: polls < 2
            ? { status: 'running', operation_id: 'op1', message: 'Apply accepted' }
            : { status: 'cancelled', operation_id: 'op1', message: 'Cancelled before the model was stopped.' },
        }
      }
      return previousGet(url)
    })
    const wrapper = await mountConfig()
    await wrapper.get('button[data-label="Restart this model"]').trigger('click')
    await flushPromises()
    await wrapper.get('.dialog button[data-label="Restart this model"]').trigger('click')
    await flushPromises()
    await wrapper.get('.dialog button[data-label="Stop apply"]').trigger('click')
    await vi.waitFor(() => expect(wrapper.vm.applyingLlamaSwap).toBe(false))
    expect(polls).toBeGreaterThan(0)
    expect(axios.post).toHaveBeenCalledWith('/api/models/model-1/runtime/apply/op1/cancel')
  })
})

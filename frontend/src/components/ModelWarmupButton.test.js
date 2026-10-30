import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axios from 'axios'
import ModelWarmupButton from './ModelWarmupButton.vue'

const { add } = vi.hoisted(() => ({ add: vi.fn() }))
vi.mock('axios', () => ({ default: { post: vi.fn() } }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add }) }))
function mountButton(model = {}, disabled = false) {
  return mount(ModelWarmupButton, {
    props: {
      model: { id: 'org/model', name: 'Demo', is_active: true, status: 'ready', ...model },
      disabled,
    },
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['disabled', 'label'],
          template: '<button :disabled="disabled">{{ label }}</button>',
        },
      },
    },
  })
}
describe('model warmup', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })
  it('uses the model probe and waits for inference success', async () => {
    let complete
    vi.mocked(axios.post).mockReturnValue(
      new Promise((resolve) => {
        complete = resolve
      }),
    )
    const wrapper = mountButton()
    await wrapper.get('button').trigger('click')
    await wrapper.get('button').trigger('click')
    expect(axios.post).toHaveBeenCalledTimes(1)
    expect(axios.post).toHaveBeenCalledWith('/api/models/org%2Fmodel/connect-test')
    expect(add).not.toHaveBeenCalled()
    complete({ data: { status_code: 200 } })
    await flushPromises()
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ severity: 'success' }))
  })
  it('reports failed inference responses as failures', async () => {
    vi.mocked(axios.post).mockResolvedValue({ data: { status_code: 503 } })
    const wrapper = mountButton()
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(add).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'error', detail: 'Inference returned HTTP 503' }),
    )
  })
  it('keeps stopped and audio models out of warmup', () => {
    for (const model of [
      { is_active: false },
      { format: 'audio_cpp' },
      { config: { engine: 'audio_cpp' } },
    ]) {
      expect(mountButton(model).find('button').exists()).toBe(false)
    }
  })
  it('disables warmup during transitions and uncertain runtime state', async () => {
    for (const model of [
      { status: 'loading' },
      { runtime_quality: 'unreachable' },
      { runtime_quality: 'stale' },
    ]) {
      const wrapper = mountButton(model)
      expect(wrapper.get('button').attributes('disabled')).toBeDefined()
      await wrapper.get('button').trigger('click')
    }
    expect(axios.post).not.toHaveBeenCalled()
  })
})

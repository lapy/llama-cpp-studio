import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { reactive, ref } from 'vue'
import SwapRoutingPanel from './SwapRoutingPanel.vue'

const toastAdd = vi.fn()
const markSwapConfigStaleLocal = vi.fn()
const fetchSwapConfigStale = vi.fn().mockResolvedValue(undefined)
const applySwapConfig = vi.fn().mockResolvedValue(undefined)
const swapConfigStale = reactive({ applicable: true, stale: false })
const fetchModels = vi.fn().mockResolvedValue(undefined)
const allQuantizations = ref([
  { llama_swap_id: 'org-model.q4_k_m', routing_name: 'friendly' },
])

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({
    markSwapConfigStaleLocal,
    fetchSwapConfigStale,
    applySwapConfig,
    swapConfigStale,
  }),
}))

vi.mock('@/stores/models', () => ({
  useModelStore: () => ({
    allQuantizations,
    fetchModels,
  }),
}))

const axiosMock = vi.hoisted(() => ({
  get: vi.fn(),
  put: vi.fn(),
  post: vi.fn(),
}))

vi.mock('axios', () => ({
  default: axiosMock,
}))

function mountPanel() {
  return mount(SwapRoutingPanel, {
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label', 'icon', 'text', 'severity', 'loading', 'outlined', 'disabled', 'rounded', 'size'],
          emits: ['click'],
          template:
            '<button :data-label="label" :aria-label="$attrs[\'aria-label\']" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
        },
        Dropdown: {
          props: ['modelValue', 'options', 'disabled'],
          emits: ['update:modelValue'],
          template:
            '<select :disabled="disabled" @change="$emit(`update:modelValue`, $event.target.value)"></select>',
        },
        InputText: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input :value="modelValue" @input="$emit(`update:modelValue`, $event.target.value)" />',
        },
        InputNumber: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input type="number" :value="modelValue" @input="$emit(`update:modelValue`, Number($event.target.value))" />',
        },
        InputSwitch: {
          props: ['modelValue'],
          emits: ['update:modelValue'],
          template:
            '<input type="checkbox" :checked="modelValue" @change="$emit(`update:modelValue`, $event.target.checked)" />',
        },
        AutoComplete: {
          props: ['modelValue', 'suggestions', 'multiple'],
          emits: ['update:modelValue', 'complete'],
          template:
            '<input :aria-label="$attrs[\'aria-label\']" :value="Array.isArray(modelValue) ? modelValue.join(\', \') : (modelValue || \'\')" @input="$emit(`update:modelValue`, multiple ? $event.target.value.split(/,\\s*/).filter(Boolean) : $event.target.value)" />',
        },
        Tag: {
          props: ['value', 'severity'],
          template: '<span>{{ value }}</span>',
        },
        Message: {
          props: ['severity'],
          template: '<div class="message-stub"><slot /></div>',
        },
      },
    },
  })
}

describe('SwapRoutingPanel', () => {
  beforeEach(() => {
    toastAdd.mockReset()
    markSwapConfigStaleLocal.mockReset()
    fetchSwapConfigStale.mockClear()
    applySwapConfig.mockClear()
    fetchModels.mockClear()
    swapConfigStale.applicable = true
    swapConfigStale.stale = false
    axiosMock.get.mockReset()
    axiosMock.put.mockReset()

    axiosMock.get.mockImplementation((url) => {
      if (url === '/api/llama-swap/routing') {
        return Promise.resolve({
          data: {
            profiles: {
              coding: { description: 'Dev', pins: { llm: 'fast' } },
            },
            selectors: {
              fast: {
                strategy: 'warm',
                targets: ['org-model.q4_k_m'],
                name: 'Fast',
                unlisted: true,
                metadata: { owner: 'studio' },
              },
            },
            warnings: ["selectors.fast.targets[0] references unknown model/alias 'maybe'"],
          },
        })
      }
      if (url === '/api/llama-swap/profiles') {
        return Promise.resolve({
          data: { active: 'coding', profiles: [{ id: 'coding' }] },
        })
      }
      return Promise.reject(new Error(`unexpected GET ${url}`))
    })
  })

  it('loads routing with Engines form patterns and validates before save', async () => {
    axiosMock.put.mockResolvedValue({
      data: {
        profiles: {
          coding: { description: 'Dev', pins: { llm: 'fast' } },
        },
        selectors: {
          fast: {
            strategy: 'warm',
            targets: ['org-model.q4_k_m'],
            name: 'Fast',
            unlisted: true,
            metadata: { owner: 'studio' },
          },
        },
        warnings: [],
        stale: true,
      },
    })

    const wrapper = mountPanel()
    await flushPromises()

    expect(axiosMock.get).toHaveBeenCalledWith('/api/llama-swap/routing')
    expect(wrapper.text()).toContain('Proxy reachable')
    expect(wrapper.text()).toContain('unknown model/alias')
    expect(wrapper.text()).toContain('Hide from')
    expect(wrapper.find('.form-row').exists()).toBe(true)
    expect(wrapper.find('.status-detail').exists()).toBe(true)

    const addSel = wrapper
      .findAll('button')
      .find((btn) => btn.attributes('data-label') === 'Add selector')
    await addSel.trigger('click')
    await wrapper.vm.save()
    await flushPromises()

    expect(axiosMock.put).not.toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'warn', summary: 'Fix routing form' })
    )

    const idInputs = wrapper
      .findAll('input')
      .filter((input) => input.element.getAttribute('aria-label') === 'Selector id')
    const targetInputs = wrapper
      .findAll('input')
      .filter((input) => input.element.getAttribute('aria-label') === 'Selector targets')
    await idInputs.at(-1).setValue('second')
    await targetInputs.at(-1).setValue('org-model.q4_k_m')
    await wrapper.vm.save()
    await flushPromises()

    expect(axiosMock.put).toHaveBeenCalledWith(
      '/api/llama-swap/routing',
      expect.objectContaining({
        profiles: expect.any(Object),
        selectors: expect.objectContaining({
          second: expect.objectContaining({
            strategy: 'warm',
            targets: ['org-model.q4_k_m'],
          }),
          fast: expect.objectContaining({
            unlisted: true,
            metadata: { owner: 'studio' },
          }),
        }),
      })
    )
    expect(markSwapConfigStaleLocal).toHaveBeenCalled()
    swapConfigStale.applicable = true
    swapConfigStale.stale = true
    await wrapper.vm.$nextTick()
    expect(wrapper.vm.showApplyLlamaSwap).toBe(true)
  })

  it('applies llama-swap config from the panel', async () => {
    swapConfigStale.applicable = true
    swapConfigStale.stale = true
    const wrapper = mountPanel()
    await flushPromises()
    await wrapper.vm.applyConfig()
    await flushPromises()
    expect(applySwapConfig).toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'llama-swap applied' })
    )
  })

  it('sets active profile when proxy is reachable', async () => {
    axiosMock.put.mockResolvedValue({ data: { active: null } })

    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.text()).toContain('Proxy reachable')

    const setBtn = wrapper.findAll('button').find((btn) => btn.attributes('data-label') === 'Set')
    await setBtn.trigger('click')
    await flushPromises()
    expect(axiosMock.put).toHaveBeenCalledWith('/api/llama-swap/profiles/active', {
      name: 'coding',
    })
  })

  it('shows proxy offline when live profiles fail', async () => {
    axiosMock.get.mockImplementation((url) => {
      if (url === '/api/llama-swap/routing') {
        return Promise.resolve({ data: { profiles: {}, selectors: {}, warnings: [] } })
      }
      if (url === '/api/llama-swap/profiles') {
        return Promise.reject(new Error('down'))
      }
      return Promise.reject(new Error(`unexpected GET ${url}`))
    })

    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.text()).toContain('Proxy offline')
    expect(wrapper.text()).toContain('Start llama-swap')
  })
})

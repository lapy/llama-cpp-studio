import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import SwapRoutingPanel from './SwapRoutingPanel.vue'

const toastAdd = vi.fn()
const markSwapConfigStaleLocal = vi.fn()

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: toastAdd }),
}))

vi.mock('@/stores/engines', () => ({
  useEnginesStore: () => ({ markSwapConfigStaleLocal }),
}))

const axiosMock = vi.hoisted(() => ({
  get: vi.fn(),
  put: vi.fn(),
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
        }),
      })
    )
    expect(markSwapConfigStaleLocal).toHaveBeenCalled()
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

import { describe, it, expect, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import ModelRow from './ModelRow.vue'

vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: () => {} }) }))

const ButtonStub = {
  props: ['label', 'icon', 'ariaLabel', 'disabled', 'loading', 'severity', 'text', 'size'],
  emits: ['click'],
  template:
    '<button :aria-label="ariaLabel" :data-icon="icon" :disabled="disabled" @click="$emit(`click`)">{{ label }}</button>',
}

function mountRow(overrides = {}) {
  return mount(ModelRow, {
    props: {
      quant: {
        id: 'model-1',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: true,
        status: 'loading',
      },
      isStarting: false,
      isStopping: false,
      formatBytes: () => '',
      formatDate: () => '',
      ...overrides,
    },
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        Button: ButtonStub,
        ModelWarmupButton: true,
        Tag: {
          props: ['value', 'title'],
          template: '<span :title="title">{{ value }}</span>',
        },
      },
    },
  })
}

function buttonByLabel(wrapper, label) {
  return wrapper.get(`button[aria-label="${label}"]`)
}

describe('ModelRow', () => {
  it('leads an unreviewed downloaded row with Set up and its Needs setup tag', async () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        quantization: 'Q4_K_M',
        is_active: false,
        status: 'stopped',
        config_reviewed: false,
      },
    })
    const setup = buttonByLabel(wrapper, 'Set up Q4_K_M')
    expect(setup.text()).toBe('Set up')
    expect(wrapper.get('.model-actions__setup-tag').text()).toBe('Needs setup')
    expect(wrapper.findAll('button[aria-label="Start Q4_K_M"]')).toHaveLength(0)

    await setup.trigger('click')
    expect(wrapper.emitted('configure')?.[0]).toEqual(['model-1'])
  })

  it('returns to the Start primary once the configuration is reviewed', async () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        quantization: 'Q4_K_M',
        is_active: false,
        status: 'stopped',
        config_reviewed: true,
      },
    })
    expect(buttonByLabel(wrapper, 'Start Q4_K_M')).toBeTruthy()
    expect(wrapper.find('.model-actions__setup-tag').exists()).toBe(false)
  })

  it('does not lead a running model with Set up even when it is unreviewed', () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        quantization: 'Q4_K_M',
        is_active: true,
        status: 'ready',
        config_reviewed: false,
      },
    })
    expect(buttonByLabel(wrapper, 'Connect Q4_K_M')).toBeTruthy()
    expect(wrapper.find('.model-actions__setup-tag').exists()).toBe(false)
  })

  it('reacts to proxy status updates from starting to running', async () => {
    const wrapper = mountRow()
    expect(wrapper.text()).toContain('Starting')

    await wrapper.setProps({
      quant: {
        id: 'model-1',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: true,
        status: 'ready',
      },
    })

    expect(wrapper.text()).toContain('Running')
    expect(wrapper.text()).not.toContain('Starting')
  })

  it('marks the row active when the quantization is running', () => {
    const wrapper = mountRow()
    expect(wrapper.get('.quant-row').classes()).toContain('is-active')
  })

  it('does not present an unreachable proxy as a running or verified stopped row', () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: false,
        status: null,
        runtime_quality: 'unreachable',
      },
    })
    expect(wrapper.text()).toContain('Status unavailable')
    expect(wrapper.text()).not.toContain('Running')
    expect(wrapper.get('.quant-row').classes()).not.toContain('is-active')
  })

  it('keeps stale observations distinct from a verified stopped row', () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: true,
        status: 'ready',
        runtime_quality: 'stale',
      },
    })
    expect(wrapper.text()).toContain('Status unavailable')
    expect(wrapper.get('[role="status"]').attributes('title')).toContain('Last seen running')
    expect(wrapper.get('.quant-row').classes()).toContain('is-active')
  })

  it('does not mark idle rows active', () => {
    const wrapper = mountRow({
      quant: {
        id: 'model-idle',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: false,
        status: 'stopped',
      },
    })
    expect(wrapper.get('.quant-row').classes()).not.toContain('is-active')
  })

  it('shows Start for a stopped model, Starting… plus Stop while starting, and Connect when running', async () => {
    const wrapper = mountRow({
      quant: { id: 'model-1', quantization: 'Q4_K_M', is_active: false, status: 'stopped' },
    })
    expect(buttonByLabel(wrapper, 'Start Q4_K_M')).toBeTruthy()
    expect(wrapper.findAll('button[aria-label="Stop Q4_K_M"]')).toHaveLength(0)

    await wrapper.setProps({ isStarting: true })
    await flushPromises()
    expect(buttonByLabel(wrapper, 'Q4_K_M is starting').text()).toContain('Starting…')
    expect(buttonByLabel(wrapper, 'Stop Q4_K_M')).toBeTruthy()

    await wrapper.setProps({
      isStarting: false,
      quant: { id: 'model-1', quantization: 'Q4_K_M', is_active: true, status: 'ready' },
    })
    await flushPromises()
    expect(buttonByLabel(wrapper, 'Connect Q4_K_M')).toBeTruthy()
    expect(buttonByLabel(wrapper, 'Stop Q4_K_M')).toBeTruthy()
  })

  it('shows a failed start as Retry start with the truncated error text available in full', () => {
    const message = 'CUDA out of memory while loading the model'
    const wrapper = mountRow({
      quant: {
        id: 'model-1',
        quantization: 'Q4_K_M',
        is_active: false,
        status: 'failed',
        last_error: message,
      },
    })
    expect(buttonByLabel(wrapper, 'Retry start for Q4_K_M')).toBeTruthy()
    const inlineError = wrapper.get('.model-actions__error')
    expect(inlineError.text()).toBe(message)
    expect(inlineError.attributes('title')).toBe(message)
  })

  it('emits audio for audio.cpp models', async () => {
    const wrapper = mountRow({
      quant: {
        id: 'audio/demo',
        name: 'demo',
        format: 'audio_cpp',
        config: { engine: 'audio_cpp' },
        is_active: false,
        status: 'stopped',
      },
    })

    const audioBtn = buttonByLabel(wrapper, 'Open audio for demo')
    await audioBtn.trigger('click')
    expect(wrapper.emitted('audio')?.[0]).toEqual(['audio/demo'])
  })

  it('exposes an accessible selection checkbox only when selectable', async () => {
    const wrapper = mountRow()
    expect(wrapper.find('input[type="checkbox"]').exists()).toBe(false)

    await wrapper.setProps({ selectable: true, selected: true })
    const checkbox = wrapper.get('input[type="checkbox"]')
    expect(checkbox.attributes('aria-label')).toBe('Select Q4_K_M')
    expect(checkbox.element.checked).toBe(true)

    await checkbox.setValue(false)
    expect(wrapper.emitted('select')?.[0]).toEqual(['model-1'])
  })
})

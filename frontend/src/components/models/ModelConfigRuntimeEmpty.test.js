import { enableAutoUnmount, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'

import ModelConfigRuntimeEmpty from './ModelConfigRuntimeEmpty.vue'

enableAutoUnmount(afterEach)

const ButtonStub = {
  name: 'Button',
  props: ['label', 'icon', 'size', 'severity', 'outlined', 'loading'],
  emits: ['click'],
  template:
    '<button type="button" :data-label="label" :disabled="loading" @click="$emit(`click`)">{{ label }}</button>',
}

function mountEmpty(props = {}) {
  return mount(ModelConfigRuntimeEmpty, {
    props,
    global: { stubs: { Button: ButtonStub } },
  })
}

describe('ModelConfigRuntimeEmpty', () => {
  it('explains a scan failure and shows saved values instead of a blank tab', () => {
    const wrapper = mountEmpty({
      engineLabel: 'llama.cpp',
      scanError: 'engine help timed out',
      savedEntries: [
        { key: 'ctx_size', label: 'Context size', value: 4096 },
        { key: 'flash_attn', label: 'Flash attention', value: true },
        { key: 'stop', label: 'Stop', value: ['END', 'STOP'] },
      ],
    })

    expect(wrapper.text()).toContain('No engine parameters yet')
    expect(wrapper.text()).toContain('llama.cpp')
    expect(wrapper.text()).toContain('engine help timed out')
    expect(wrapper.text()).toContain('Saved settings')
    expect(wrapper.text()).toContain('Context size')
    expect(wrapper.text()).toContain('4096')
    expect(wrapper.text()).toContain('On')
    expect(wrapper.text()).toContain('END, STOP')
  })

  it('explains the pending scan state and hides the saved section when empty', () => {
    const wrapper = mountEmpty({
      engineLabel: 'llama.cpp',
      scanPending: true,
      savedEntries: [],
    })

    expect(wrapper.text()).toContain('still loading')
    expect(wrapper.find('.runtime-empty__saved').exists()).toBe(false)
  })

  it('offers rescan and Open Engines actions', async () => {
    const wrapper = mountEmpty({ engineLabel: 'llama.cpp' })

    await wrapper.get('[data-runtime-rescan]').trigger('click')
    await wrapper.get('[data-runtime-open-engines]').trigger('click')

    expect(wrapper.emitted('rescan')).toHaveLength(1)
    expect(wrapper.emitted('open-engines')).toHaveLength(1)
  })

  it('disables rescan while a scan is in flight', () => {
    const wrapper = mountEmpty({ rescanning: true })

    expect(wrapper.get('[data-runtime-rescan]').element.disabled).toBe(true)
  })
})

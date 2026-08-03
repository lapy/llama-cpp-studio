import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'

import ModelRow from './ModelRow.vue'

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
        Button: true,
        ModelStartStopButton: true,
        Tag: {
          props: ['value'],
          template: '<span>{{ value }}</span>',
        },
      },
    },
  })
}

describe('ModelRow', () => {
  it('reacts to proxy status updates from loading to ready', async () => {
    const wrapper = mountRow()
    expect(wrapper.text()).toContain('Loading')

    await wrapper.setProps({
      quant: {
        id: 'model-1',
        name: 'Q4_K_M',
        quantization: 'Q4_K_M',
        is_active: true,
        status: 'ready',
      },
    })

    expect(wrapper.text()).toContain('Ready')
    expect(wrapper.text()).not.toContain('Loading')
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

    const audioBtn = wrapper
      .findAllComponents({ name: 'Button' })
      .find((btn) => btn.attributes('icon') === 'pi pi-volume-up')
    expect(audioBtn).toBeTruthy()
    await audioBtn.trigger('click')
    expect(wrapper.emitted('audio')?.[0]).toEqual(['audio/demo'])
  })
})

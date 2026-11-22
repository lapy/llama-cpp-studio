import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ModelConfigHeader from './ModelConfigHeader.vue'

const model = {
  display_name: 'Demo Model',
  base_model_name: 'demo',
  format: 'gguf',
  quantization: 'Q4_K_M',
  family: 'llama',
  tasks: ['text-generation'],
  huggingface_id: 'example/demo',
}

function mountHeader(props = {}) {
  return mount(ModelConfigHeader, {
    props: { model, ...props },
    global: {
      stubs: {
        Button: {
          props: ['label', 'ariaLabel'],
          emits: ['click'],
          template: '<button :aria-label="ariaLabel" @click="$emit(`click`)">{{ label }}</button>',
        },
        Tag: { props: ['value'], template: '<span>{{ value }}</span>' },
      },
    },
  })
}

describe('ModelConfigHeader', () => {
  it('renders model identity metadata and optional actions', () => {
    const wrapper = mountHeader({
      canCheckForUpdates: true,
      isAudioEngine: true,
    })

    expect(wrapper.get('h1').text()).toBe('Demo Model')
    expect(wrapper.text()).toContain('Review the engine, runtime parameters, names, and launch')
    expect(wrapper.text()).toContain('Demo Model')
    expect(wrapper.text()).toContain('GGUF')
    expect(wrapper.text()).toContain('Q4_K_M')
    expect(wrapper.text()).toContain('Text generation')
    expect(wrapper.get('a').attributes('href')).toBe('https://huggingface.co/example/demo')
    expect(wrapper.get('a').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.text()).toContain('Check for updates')
    expect(wrapper.text()).toContain('Open Audio')
  })

  it('forwards navigation and action intent to the view', async () => {
    const wrapper = mountHeader({ canCheckForUpdates: true, isAudioEngine: true })
    const buttons = wrapper.findAll('button')

    await buttons[0].trigger('click')
    await buttons[1].trigger('click')
    await buttons[2].trigger('click')

    expect(wrapper.emitted('back')).toHaveLength(1)
    expect(wrapper.emitted('check-for-updates')).toHaveLength(1)
    expect(wrapper.emitted('open-audio')).toHaveLength(1)
  })
})

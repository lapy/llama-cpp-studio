import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ModelConfigAutoFit from './ModelConfigAutoFit.vue'

const defaults = { fit: 'on', fitCtx: 4096, fitTarget: 1024 }

const overrides = [
  {
    key: 'fit',
    label: 'Auto-fit',
    current: 'off',
    autoLabel: 'On',
    restore: { mode: 'set', value: 'on' },
    reason:
      'Auto-fit is explicitly off, so the engine keeps fixed context and layer settings instead of sizing them to memory.',
  },
  {
    key: 'ctx_size',
    label: 'Context size',
    current: 8192,
    autoLabel: 'Auto (0)',
    restore: { mode: 'set', value: 0 },
    reason:
      'An explicit --ctx-size prevents auto-fit from choosing the context. Setting 0 loads the model context instead.',
  },
  {
    key: 'n_gpu_layers',
    label: 'GPU layers',
    current: 32,
    autoLabel: 'Auto',
    restore: { mode: 'clear' },
    reason:
      'A numeric layer count prevents auto-fit from choosing layers. Clearing it lets the engine choose per-device layers.',
  },
]

function mountCard(props = {}) {
  return mount(ModelConfigAutoFit, {
    props: { active: false, overrides, defaults, ...props },
    global: {
      stubs: {
        Button: {
          props: ['label'],
          emits: ['click'],
          template: '<button :data-label="label" @click="$emit(`click`)">{{ label }}</button>',
        },
      },
    },
  })
}

describe('ModelConfigAutoFit', () => {
  it('explains the active auto-fit state without any action or fabricated numbers', () => {
    const wrapper = mountCard({ active: true, overrides: [] })

    expect(wrapper.get('h3').text()).toBe('Auto-fit is active')
    expect(wrapper.text()).toContain('--fit')
    expect(wrapper.text()).toContain('Engine defaults: --fit on · --fit-ctx 4,096 tokens')
    expect(wrapper.text()).toContain('--fit-target 1,024 MiB')
    expect(wrapper.find('ul').exists()).toBe(false)
    expect(wrapper.find('[data-auto-fit-restore]').exists()).toBe(false)
    expect(wrapper.text()).not.toMatch(/VRAM|GiB|recommended for this machine/i)
  })

  it('lists each override with its current value, auto target, and reason', () => {
    const wrapper = mountCard()

    expect(wrapper.get('h3').text()).toBe('Auto-fit is overridden')
    expect(wrapper.get('ul').attributes('aria-label')).toBe('Settings overriding auto-fit')
    expect(wrapper.findAll('[data-auto-fit-key]')).toHaveLength(3)

    const rows = wrapper.findAll('[data-auto-fit-key]')
    expect(rows[0].text()).toContain('Auto-fit')
    expect(rows[0].text()).toContain('Off')
    expect(rows[0].text()).toContain('On')
    expect(rows[1].text()).toContain('Context size')
    expect(rows[1].text()).toContain('8,192 tokens')
    expect(rows[1].text()).toContain('Auto (0)')
    expect(rows[1].text()).toContain('explicit --ctx-size')
    expect(rows[2].text()).toContain('32 layers')
    expect(rows[2].text()).toContain('Auto')
    expect(wrapper.text()).toContain('3 overrides')
  })

  it('emits restore and review intents', async () => {
    const wrapper = mountCard()

    await wrapper.get('[data-auto-fit-restore]').trigger('click')
    await wrapper.get('[data-auto-fit-review]').trigger('click')

    expect(wrapper.emitted('restore')).toHaveLength(1)
    expect(wrapper.emitted('review')).toHaveLength(1)
  })
})

import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ModelConfigGpuCard from './ModelConfigGpuCard.vue'

const gpus = [
  { value: 'GPU-0', name: 'RTX 4090', index: 0 },
  { value: 'GPU-1', name: 'RTX 3090', index: 1 },
]

function mountCard(props = {}) {
  return mount(ModelConfigGpuCard, {
    props: { gpus, ...props },
    global: {
      directives: { tooltip: () => {} },
    },
  })
}

describe('ModelConfigGpuCard', () => {
  it('renders the assignment modes and detected devices', () => {
    const wrapper = mountCard()

    const radios = wrapper.findAll('#gpu-mode [role="radio"]')
    expect(radios.map((radio) => radio.text())).toEqual(['All GPUs', 'Choose GPUs', 'CPU only'])
    expect(radios[0].attributes('aria-checked')).toBe('true')
    expect(radios[0].attributes('tabindex')).toBe('0')
    expect(radios[1].attributes('aria-checked')).toBe('false')
    expect(radios[1].attributes('tabindex')).toBe('-1')
    expect(wrapper.text()).toContain('GPUs')
    expect(wrapper.find('.gpu-card-grid').exists()).toBe(false)
  })

  it('keeps the card classes passed by its host', () => {
    const wrapper = mountCard({ class: 'config-card' })

    expect(wrapper.classes()).toContain('gpu-bind')
    expect(wrapper.classes()).toContain('config-card')
  })

  it('shows the device grid in selected mode and emits the chosen mode', async () => {
    const wrapper = mountCard({ mode: 'selected' })

    expect(wrapper.findAll('.gpu-card')).toHaveLength(2)
    expect(wrapper.get('.gpu-card').text()).toContain('RTX 4090')
    expect(wrapper.get('.gpu-card').text()).toContain('Not used')

    await wrapper.findAll('#gpu-mode button')[1].trigger('click')
    expect(wrapper.emitted('update:mode')).toEqual([['selected']])
  })

  it('moves the mode selection with arrow keys', async () => {
    const wrapper = mountCard()

    await wrapper.get('#gpu-mode').trigger('keydown', { key: 'ArrowRight' })
    expect(wrapper.emitted('update:mode')).toEqual([['selected']])
  })

  it('emits the chosen device order when cards are toggled', async () => {
    const wrapper = mountCard({ mode: 'selected', selected: ['GPU-1'] })
    const cards = wrapper.findAll('.gpu-card')
    expect(cards[1].text()).toContain('Device 0')

    await cards[0].trigger('click')
    expect(wrapper.emitted('update:selected')).toEqual([[['GPU-1', 'GPU-0']]])

    const ordered = mountCard({ mode: 'selected', selected: ['GPU-0', 'GPU-1'] })
    expect(ordered.findAll('.gpu-card')[1].text()).toContain('Device 1')
    await ordered.findAll('.gpu-card')[0].trigger('click')
    expect(ordered.emitted('update:selected')).toEqual([[['GPU-1']]])
  })

  it('renders no device cards when none are detected', () => {
    const wrapper = mountCard({ mode: 'selected', gpus: [] })

    expect(wrapper.find('.gpu-card-grid').exists()).toBe(true)
    expect(wrapper.findAll('.gpu-card')).toHaveLength(0)
  })

  it('blocks every change while disabled', async () => {
    const wrapper = mountCard({ mode: 'selected', selected: [], disabled: true })

    expect(
      wrapper
        .findAll('#gpu-mode button')
        .every((button) => button.attributes('disabled') !== undefined),
    ).toBe(true)
    expect(
      wrapper.findAll('.gpu-card').every((card) => card.attributes('disabled') !== undefined),
    ).toBe(true)

    await wrapper.findAll('#gpu-mode button')[1].trigger('click')
    await wrapper.findAll('.gpu-card')[0].trigger('click')
    await wrapper.get('#gpu-mode').trigger('keydown', { key: 'ArrowRight' })

    expect(wrapper.emitted('update:mode')).toBeUndefined()
    expect(wrapper.emitted('update:selected')).toBeUndefined()
  })
})

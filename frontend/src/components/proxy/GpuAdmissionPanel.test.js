import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import GpuAdmissionPanel from './GpuAdmissionPanel.vue'

const Tag = {
  props: ['value', 'severity'],
  template: '<span class="tag" :data-severity="severity">{{ value }}</span>',
}

const Button = {
  props: ['label', 'icon', 'ariaLabel'],
  emits: ['click'],
  template: '<button :aria-label="ariaLabel" @click="$emit(\'click\')">{{ label }}</button>',
}

function snapshot(overrides = {}) {
  return {
    mode: 'warn',
    safety_margin_mib: 2048,
    estimates_configured: { demo: 8192 },
    observed_estimates: { demo: { mib: 8000, at: '2026-01-01T00:00:00Z', devices: [0] } },
    probe: {
      status: 'ok',
      vendor: 'nvidia',
      at: '2026-01-01T00:00:00Z',
      devices: [{ index: 0, free_mib: 8192 }],
      error: null,
    },
    last_admission: {
      decision: 'rejected',
      model: 'demo',
      estimate_mib: 16384,
      estimate_source: 'settings',
      available_mib: 7168,
      deficit_mib: 9216,
      reason: 'Estimate 16384 MiB (settings) exceeds 7168 MiB available',
    },
    ...overrides,
  }
}

function mountPanel(props = {}) {
  return mount(GpuAdmissionPanel, {
    props: {
      mode: 'warn',
      margin: 2048,
      rows: [{ model: 'demo', mib: 8192 }],
      catalog: [
        { id: 'demo', studio_model_id: 'stored-demo' },
        { id: 'other', studio_model_id: 'stored-other' },
      ],
      snapshot: snapshot(),
      ...props,
    },
    global: { stubs: { Tag, Button } },
  })
}

describe('GpuAdmissionPanel', () => {
  it('shows the probe state and the last admission decision with its evidence', () => {
    const wrapper = mountPanel()
    expect(wrapper.get('#proxy-vram-title').text()).toBe('GPU admission')
    expect(wrapper.text()).toContain('Probe ready')
    expect(wrapper.text()).toContain('demo · 8,000 MiB')
    expect(wrapper.text()).toContain('16,384 MiB settings')
    expect(wrapper.text()).toContain('9,216 MiB short')
    expect(wrapper.text()).toContain('exceeds 7168 MiB available')
    const tags = wrapper.findAll('.tag')
    expect(tags[0].attributes('data-severity')).toBe('success')
    expect(tags[0].text()).toBe('Probe ready')
    expect(tags[tags.length - 1].attributes('data-severity')).toBe('danger')
    expect(tags[tags.length - 1].text()).toBe('rejected')
  })

  it('reports an unprobed CPU deployment without inventing a decision', () => {
    const wrapper = mountPanel({
      snapshot: {
        mode: 'off',
        probe: { status: 'never' },
        observed_estimates: {},
        last_admission: null,
      },
    })
    expect(wrapper.text()).toContain('Probe not run yet')
    expect(wrapper.text()).toContain('No uncontended load observed yet.')
    expect(wrapper.text()).toContain('No admission decision recorded yet.')
  })

  it('emits live mode and margin changes', async () => {
    const wrapper = mountPanel()
    await wrapper.get('select[aria-label="VRAM admission mode"]').setValue('enforce')
    expect(wrapper.emitted('update:mode')).toEqual([['enforce']])
    const margin = wrapper.get('input[aria-label="VRAM safety margin MiB"]')
    await margin.setValue('3072')
    expect(wrapper.emitted('update:margin')).toEqual([[3072]])
    await margin.setValue('')
    expect(wrapper.emitted('update:margin')[1]).toEqual([null])
  })

  it('edits estimate rows through emitted copies', async () => {
    const wrapper = mountPanel()
    await wrapper.get('#vram-model-0').setValue('other')
    expect(wrapper.emitted('update:rows')[0][0]).toEqual([{ model: 'other', mib: 8192 }])

    await wrapper.get('#vram-mib-0').setValue('4096')
    expect(wrapper.emitted('update:rows')[1][0]).toEqual([{ model: 'demo', mib: 4096 }])

    await wrapper.get('button[aria-label="Remove VRAM estimate"]').trigger('click')
    expect(wrapper.emitted('update:rows')[2][0]).toEqual([])

    const addButton = wrapper.findAll('button').find((button) => button.text() === 'Add estimate')
    await addButton.trigger('click')
    expect(wrapper.emitted('update:rows')[3][0]).toEqual([
      { model: 'demo', mib: 8192 },
      { model: '', mib: null },
    ])
  })

  it('keeps a configured model selectable when it is not in the catalog', () => {
    const wrapper = mountPanel({ rows: [{ model: 'retired-model', mib: 4096 }] })
    const options = wrapper
      .get('#vram-model-0')
      .findAll('option')
      .map((option) => option.text())
    expect(options).toContain('retired-model')
  })

  it('labels every control for assistive technology', () => {
    const wrapper = mountPanel()
    expect(wrapper.get('select[aria-label="VRAM admission mode"]').exists()).toBe(true)
    expect(wrapper.get('input[aria-label="VRAM safety margin MiB"]').exists()).toBe(true)
    expect(wrapper.get('#vram-model-0').attributes('id')).toBe('vram-model-0')
    expect(wrapper.get('button[aria-label="Remove VRAM estimate"]').exists()).toBe(true)
    expect(wrapper.get('[role="status"] .vram-state').exists()).toBe(true)
  })
})

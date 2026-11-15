import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import RequestInspectorPanel from './RequestInspectorPanel.vue'

const Dialog = {
  props: ['visible', 'header', 'modal', 'style'],
  emits: ['update:visible'],
  template:
    '<div v-if="visible" role="dialog" :data-header="header"><button class="dialog-close" @click="$emit(`update:visible`, false)">close</button><slot /></div>',
}

const Tag = {
  props: ['value', 'severity'],
  template: '<span class="tag" :data-severity="severity">{{ value }}</span>',
}

function request(overrides = {}) {
  return {
    id: 'r1',
    state: 'completed',
    status: 200,
    method: 'POST',
    path: '/v1/chat/completions',
    model: 'demo',
    latency_ms: 12,
    started_at: 1700000000,
    output_tokens: 4,
    output_tokens_per_second: 4,
    engine_build: JSON.stringify({ version: 'b1234', source_commit: 'abcdef1234567890' }),
    ...overrides,
  }
}

function mountPanel(selected = request(), visible = true) {
  return mount(RequestInspectorPanel, {
    props: { request: selected, visible },
    global: { stubs: { Dialog, Tag } },
  })
}

function detailValue(wrapper, label) {
  const row = wrapper
    .findAll('.request-details div')
    .find((candidate) => candidate.get('dt').text() === label)
  return row?.get('dd').text()
}

describe('RequestInspectorPanel', () => {
  it('renders the selected request details and reveals unavailable fields on demand', async () => {
    const wrapper = mountPanel()
    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.attributes('data-header')).toBe('Request details')
    expect(wrapper.get('.tag').attributes('data-severity')).toBe('success')
    expect(wrapper.get('.tag').text()).toBe('completed')

    const inspector = wrapper.get('.request-inspector')
    expect(inspector.text()).toContain('POST')
    expect(inspector.text()).toContain('/v1/chat/completions')
    expect(inspector.text()).not.toContain('Time to first byte')
    expect(detailValue(wrapper, 'Total duration')).toBe('12.0 ms')
    expect(detailValue(wrapper, 'Output tokens per second (legacy alias)')).toBe('4.00 tok/s')
    expect(detailValue(wrapper, 'Installed engine build identity')).toBe(
      'b1234 · commit abcdef1234567890',
    )
    expect(detailValue(wrapper, 'Started')).toBe(new Date(1700000000 * 1000).toLocaleString())

    await wrapper.get('.inspector-field-toggle input').setValue(true)
    expect(detailValue(wrapper, 'Time to first byte')).toBe('—')
    wrapper.unmount()
  })

  it('emits close and resets the unavailable-fields toggle when the request changes', async () => {
    const wrapper = mountPanel()
    await wrapper.get('.inspector-field-toggle input').setValue(true)
    expect(wrapper.get('.request-inspector').text()).toContain('Time to first byte')

    await wrapper.get('.dialog-close').trigger('click')
    expect(wrapper.emitted('update:visible')).toEqual([[false]])

    await wrapper.setProps({ request: request({ id: 'r2', ttft_ms: null }) })
    expect(wrapper.get('.inspector-field-toggle input').element.checked).toBe(false)
    expect(wrapper.get('.request-inspector').text()).not.toContain('Time to first byte')

    await wrapper.setProps({ visible: false })
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('renders booleans and missing values consistently', () => {
    const wrapper = mountPanel(
      request({ nonstream_response: false, usage_source: '', response_transport: null }),
    )
    expect(detailValue(wrapper, 'Non-streaming response')).toBe('No')
    expect(detailValue(wrapper, 'Usage source')).toBe('—')
    expect(wrapper.get('.request-inspector').text()).not.toContain('Response transport')
    wrapper.unmount()
  })
})

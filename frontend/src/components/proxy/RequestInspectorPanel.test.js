import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import RequestInspectorPanel from './RequestInspectorPanel.vue'
import axios from 'axios'
import { copyText } from '@/utils/clipboard'
import {
  buildCurlCommand,
  buildHarExport,
  buildJsonExport,
  downloadTextFile,
} from '@/utils/proxyRequestExport'

const toastAdd = vi.fn()
vi.mock('axios', () => ({ default: { post: vi.fn() } }))
vi.mock('primevue/usetoast', () => ({ useToast: () => ({ add: toastAdd }) }))
vi.mock('@/utils/clipboard', () => ({ copyText: vi.fn() }))
vi.mock('@/utils/proxyRequestExport', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    buildCurlCommand: vi.fn(actual.buildCurlCommand),
    buildJsonExport: vi.fn(actual.buildJsonExport),
    buildHarExport: vi.fn(actual.buildHarExport),
    downloadTextFile: vi.fn(),
  }
})

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
    attachTo: document.body,
    global: { stubs: { Dialog, Tag } },
  })
}

function detailValue(wrapper, label) {
  const row = wrapper
    .findAll('.request-details div')
    .find((candidate) => candidate.get('dt').text() === label)
  return row?.get('dd').text()
}

async function openExportMenu(wrapper) {
  await wrapper.get('.request-export-toggle').trigger('click')
  await flushPromises()
  return wrapper.findAll('[role="menuitem"]')
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

describe('RequestInspectorPanel export actions', () => {
  beforeEach(() => {
    toastAdd.mockClear()
    copyText.mockReset()
    copyText.mockResolvedValue(undefined)
    buildCurlCommand.mockClear()
    buildJsonExport.mockClear()
    buildHarExport.mockClear()
    downloadTextFile.mockClear()
  })

  it('copies a cURL command built from the captured request', async () => {
    const selected = request({
      headers: { 'content-type': 'application/json', authorization: "Bearer a'b" },
      body: '{"model":"demo"}',
    })
    const wrapper = mountPanel(selected)
    const toggle = wrapper.get('.request-export-toggle')
    expect(toggle.attributes('aria-expanded')).toBe('false')

    await openExportMenu(wrapper)
    expect(toggle.attributes('aria-expanded')).toBe('true')
    expect(wrapper.findAll('[role="menuitem"]').map((item) => item.text())).toEqual([
      'Copy as cURL',
      'Download JSON',
      'Download HAR',
    ])
    expect(wrapper.find('.request-export-note').exists()).toBe(false)

    await wrapper.get('[data-export-action="curl"]').trigger('click')
    await flushPromises()

    expect(buildCurlCommand).toHaveBeenCalledWith(expect.objectContaining({ id: 'r1' }), {
      origin: window.location.origin,
    })
    expect(copyText).toHaveBeenCalledWith(
      expect.stringContaining(`curl --request POST '${window.location.origin}`),
    )
    expect(copyText.mock.calls[0][0]).toContain(`--header 'authorization: Bearer a'"'"'b'`)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({ severity: 'success', summary: 'cURL command copied' }),
    )
    expect(wrapper.find('[role="menu"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('downloads pretty-printed JSON and a valid HAR with safe filenames', async () => {
    const wrapper = mountPanel(request({ id: 'r1' }))

    await openExportMenu(wrapper)
    await wrapper.get('[data-export-action="json"]').trigger('click')
    await flushPromises()

    expect(buildJsonExport).toHaveBeenCalledWith(expect.objectContaining({ id: 'r1' }))
    expect(downloadTextFile).toHaveBeenCalledWith(
      expect.stringContaining('"id": "r1"'),
      'request-r1.json',
      'application/json',
    )
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'JSON downloaded',
        detail: 'request-r1.json',
      }),
    )

    await openExportMenu(wrapper)
    await wrapper.get('[data-export-action="har"]').trigger('click')
    await flushPromises()

    expect(buildHarExport).toHaveBeenCalledWith(expect.objectContaining({ id: 'r1' }), {
      origin: window.location.origin,
      appVersion: expect.any(String),
    })
    const [harContent, harFilename, harMime] = downloadTextFile.mock.calls[1]
    expect(harFilename).toBe('request-r1.har')
    expect(harMime).toBe('application/json')
    const har = JSON.parse(harContent)
    expect(har.log.version).toBe('1.2')
    expect(har.log.entries).toHaveLength(1)
    expect(har.log.entries[0].request.url).toBe(`${window.location.origin}/v1/chat/completions`)
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'HAR downloaded',
        detail: 'request-r1.har',
      }),
    )
    wrapper.unmount()
  })

  it('labels cURL export when headers and bodies were not captured', async () => {
    const wrapper = mountPanel(request())

    await openExportMenu(wrapper)
    expect(wrapper.get('[data-export-action="curl"]').attributes('title')).toContain(
      'headers and body were not captured',
    )
    expect(wrapper.get('.request-export-note').text()).toContain(
      'Headers and bodies are not retained',
    )

    await wrapper.get('[data-export-action="curl"]').trigger('click')
    await flushPromises()

    expect(copyText).toHaveBeenCalledWith(
      `curl --request POST '${window.location.origin}/v1/chat/completions'`,
    )
    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'success',
        summary: 'cURL command copied',
        detail: expect.stringContaining('not captured'),
      }),
    )
    wrapper.unmount()
  })

  it('supports arrow keys, End, and Escape in the export menu', async () => {
    const wrapper = mountPanel(request())
    const toggle = wrapper.get('.request-export-toggle')

    await toggle.trigger('click')
    await flushPromises()
    const items = wrapper.findAll('[role="menuitem"]')
    expect(document.activeElement).toBe(items[0].element)

    await wrapper.get('[role="menu"]').trigger('keydown', { key: 'ArrowDown' })
    expect(document.activeElement).toBe(items[1].element)
    await wrapper.get('[role="menu"]').trigger('keydown', { key: 'End' })
    expect(document.activeElement).toBe(items[2].element)
    await wrapper.get('[role="menu"]').trigger('keydown', { key: 'ArrowDown' })
    expect(document.activeElement).toBe(items[0].element)

    await wrapper.get('[role="menu"]').trigger('keydown', { key: 'Escape' })
    await flushPromises()
    expect(wrapper.find('[role="menu"]').exists()).toBe(false)
    expect(document.activeElement).toBe(toggle.element)
    wrapper.unmount()
  })

  it('replays a captured request and shows the engine response', async () => {
    vi.mocked(axios.post).mockResolvedValue({
      data: { status: 200, body: '{"ok":true}', body_truncated: false },
    })
    const wrapper = mountPanel(
      request({
        replayable: true,
        captured_body: '{"messages":[]}',
        captured_body_truncated: false,
        captured_redacted_headers: ['authorization'],
      }),
    )

    expect(wrapper.get('[data-captured-body]').text()).toContain('{"messages":[]}')
    expect(wrapper.text()).toContain('Redacted headers: authorization')

    await wrapper.get('[data-request-replay]').trigger('click')
    await flushPromises()

    expect(axios.post).toHaveBeenCalledWith('/api/proxy/requests/r1/replay')
    expect(wrapper.get('[data-replay-result]').text()).toContain('200')
    expect(wrapper.get('[data-replay-result]').text()).toContain('{"ok":true}')
    wrapper.unmount()
  })

  it('shows the replay error message without claiming success', async () => {
    vi.mocked(axios.post).mockRejectedValue({
      response: { data: { detail: { message: 'Start demo before replaying this request.' } } },
    })
    const wrapper = mountPanel(request({ replayable: true }))

    await wrapper.get('[data-request-replay]').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('Start demo before replaying this request.')
    expect(wrapper.find('[data-replay-result]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('hides the replay control when the payload was not captured', () => {
    const wrapper = mountPanel(request())
    expect(wrapper.find('[data-request-replay]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('reports a copy failure without claiming success', async () => {
    copyText.mockRejectedValueOnce(new Error('Clipboard denied'))
    const wrapper = mountPanel(request())

    await openExportMenu(wrapper)
    await wrapper.get('[data-export-action="curl"]').trigger('click')
    await flushPromises()

    expect(toastAdd).toHaveBeenCalledWith(
      expect.objectContaining({
        severity: 'error',
        summary: 'Could not copy the cURL command',
        detail: 'Clipboard denied',
      }),
    )
    expect(toastAdd).not.toHaveBeenCalledWith(
      expect.objectContaining({ summary: 'cURL command copied' }),
    )
    wrapper.unmount()
  })
})

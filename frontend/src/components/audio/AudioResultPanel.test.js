import { afterEach, describe, expect, it } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'

import AudioResultPanel from './AudioResultPanel.vue'

enableAutoUnmount(afterEach)

// No emits:['click'] so parent @click falls through as a native listener.
const buttonStub = {
  props: ['label', 'icon', 'size', 'severity', 'outlined'],
  template: '<button type="button" :data-label="label">{{ label }}</button>',
}

function mountPanel(props = {}) {
  return mount(AudioResultPanel, {
    props,
    global: {
      stubs: {
        Button: buttonStub,
        Message: {
          props: ['severity', 'closable'],
          template: '<div class="message"><slot /></div>',
        },
      },
    },
  })
}

describe('AudioResultPanel', () => {
  it('renders nothing when empty', () => {
    const wrapper = mountPanel()
    expect(wrapper.find('.config-card').exists()).toBe(false)
  })

  it('renders player and download for each audio clip', async () => {
    const clips = [
      {
        id: 'audio',
        label: 'Audio',
        url: 'blob:primary',
        blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
        filename: 'music.wav',
      },
      {
        id: 'drums',
        label: 'drums',
        url: 'blob:drums',
        blob: new Blob([new Uint8Array([2])], { type: 'audio/wav' }),
        filename: 'drums.wav',
      },
    ]
    const wrapper = mountPanel({ clips })

    const players = wrapper.findAll('audio.audio-player')
    expect(players).toHaveLength(2)
    expect(players[0].attributes('src')).toBe('blob:primary')
    expect(players[1].attributes('src')).toBe('blob:drums')
    expect(wrapper.text()).toContain('drums')
    expect(wrapper.find('button[data-label="Download Audio"]').exists()).toBe(true)
    expect(wrapper.find('button[data-label="Download drums"]').exists()).toBe(true)

    await wrapper.find('button[data-label="Download Audio"]').trigger('click')
    await flushPromises()
    expect(wrapper.emitted('download')?.[0]?.[0]).toMatchObject({
      id: 'audio',
      filename: 'music.wav',
    })
  })

  it('shows error and metadata text alongside clips', () => {
    const wrapper = mountPanel({
      error: 'boom',
      text: '{"timing":{"wall_ms":12}}',
      clips: [
        {
          id: 'audio',
          label: 'Audio',
          url: 'blob:a',
          blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
          filename: 'audio.wav',
        },
      ],
    })

    expect(wrapper.text()).toContain('boom')
    expect(wrapper.text()).toContain('"wall_ms":12')
    expect(wrapper.find('audio').exists()).toBe(true)
  })

  it('uses Download WAV label for a single clip', () => {
    const wrapper = mountPanel({
      clips: [
        {
          id: 'audio',
          label: 'Audio',
          url: 'blob:a',
          blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }),
          filename: 'speech.wav',
        },
      ],
    })
    expect(wrapper.find('button[data-label="Download WAV"]').exists()).toBe(true)
  })
})

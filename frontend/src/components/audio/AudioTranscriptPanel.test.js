import { afterEach, describe, expect, it } from 'vitest'
import { enableAutoUnmount, mount } from '@vue/test-utils'

import AudioTranscriptPanel from './AudioTranscriptPanel.vue'

enableAutoUnmount(afterEach)

function mountPanel(props = {}) {
  return mount(AudioTranscriptPanel, { props })
}

describe('AudioTranscriptPanel', () => {
  it('renders nothing before a transcript exists', () => {
    const wrapper = mountPanel()
    expect(wrapper.find('.page-section').exists()).toBe(false)
    expect(wrapper.get('.audio-result-region').attributes('aria-busy')).toBe('false')
  })

  it('announces the loading state through the shared result region', () => {
    const wrapper = mountPanel({ loading: true })
    expect(wrapper.get('.audio-result-region').attributes('role')).toBe('status')
    expect(wrapper.get('.audio-result-region').attributes('aria-live')).toBe('polite')
    expect(wrapper.get('.audio-result-region').attributes('aria-busy')).toBe('true')
  })

  it('shows a shared section heading, error banner and transcript text', () => {
    const wrapper = mountPanel({
      error: 'Transcription failed',
      text: 'hello world',
      metadata: '{"segments":1}',
    })
    expect(wrapper.get('.page-section__title').text()).toBe('Transcript')
    expect(wrapper.get('.state-banner--error').text()).toContain('Transcription failed')
    expect(wrapper.text()).toContain('hello world')
    expect(wrapper.text()).toContain('"segments":1')
  })

  it('lists each batch result with its filename', () => {
    const wrapper = mountPanel({
      batchResults: [
        { name: 'first.wav', text: 'first transcript' },
        { name: 'second.wav', text: 'second transcript' },
      ],
    })
    const items = wrapper.findAll('.asr-batch-results li')
    expect(items).toHaveLength(2)
    expect(items[0].text()).toContain('first.wav')
    expect(items[1].text()).toContain('second transcript')
  })
})

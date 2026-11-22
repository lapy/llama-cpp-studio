import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import HuggingFaceTokenNotice from './HuggingFaceTokenNotice.vue'

describe('HuggingFaceTokenNotice', () => {
  it('renders the shared info banner and emits the next actions', async () => {
    const wrapper = mount(HuggingFaceTokenNotice, {
      props: { dismissible: true },
    })

    expect(wrapper.get('.huggingface-token-notice').classes()).toContain('state-banner--info')
    expect(wrapper.text()).toContain('A Hugging Face token is required for gated models.')
    expect(wrapper.text()).toContain('Set token')
    expect(wrapper.text()).toContain('Dismiss')

    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Set token')
      .trigger('click')
    await wrapper
      .findAll('button')
      .find((button) => button.text() === 'Dismiss')
      .trigger('click')

    expect(wrapper.emitted('open')).toHaveLength(1)
    expect(wrapper.emitted('dismiss')).toHaveLength(1)
    wrapper.unmount()
  })

  it('omits the dismiss control when not dismissible', () => {
    const wrapper = mount(HuggingFaceTokenNotice)
    expect(wrapper.text()).not.toContain('Dismiss')
    wrapper.unmount()
  })
})

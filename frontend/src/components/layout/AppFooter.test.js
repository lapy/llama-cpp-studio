import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { reactive, nextTick } from 'vue'

const progressStore = reactive({
  isConnected: true,
})

vi.mock('@/stores/progress', () => ({
  useProgressStore: () => progressStore,
}))

import AppFooter from './AppFooter.vue'

describe('AppFooter', () => {
  it('shows live status and updates when the SSE connection drops', async () => {
    const wrapper = mount(AppFooter)

    expect(wrapper.text()).toContain('llama.cpp Studio v')
    expect(wrapper.text()).toContain('Live')
    expect(wrapper.find('.footer-status--ok').exists()).toBe(true)

    progressStore.isConnected = false
    await nextTick()

    expect(wrapper.text()).toContain('Reconnecting…')
    expect(wrapper.find('.footer-status--warn').exists()).toBe(true)
  })
})

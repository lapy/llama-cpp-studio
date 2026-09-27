import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'

import AppHeader from './AppHeader.vue'

function mountHeader(props = {}) {
  return mount(AppHeader, {
    props,
    global: {
      directives: {
        tooltip: () => {},
      },
      stubs: {
        ThemeToggle: { template: '<div class="theme-toggle-stub" />' },
        SwapConfigHeaderNotice: { template: '<div class="swap-notice-stub" />' },
      },
    },
  })
}

describe('AppHeader', () => {
  it('shows a healthy llama-swap indicator and links to the configured UI port', () => {
    const wrapper = mountHeader({ llamaSwapStatus: { healthy: true, port: 2345 } })

    expect(wrapper.get('.status-light').classes()).toContain('status-light--online')
    expect(wrapper.get('a.llama-swap-link').attributes('href')).toContain(':2345/ui')
    expect(wrapper.text()).toContain('llama-swap')
  })

  it('keeps proxy status unknown until a health value arrives', async () => {
    const wrapper = mountHeader({ llamaSwapStatus: null })

    expect(wrapper.get('.status-light').classes()).toContain('status-light--unknown')
    expect(wrapper.get('a.llama-swap-link').attributes('aria-label')).toContain('unknown')

    await wrapper.setProps({ llamaSwapStatus: { healthy: false } })
    expect(wrapper.get('.status-light').classes()).toContain('status-light--offline')
  })
})

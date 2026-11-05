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
        RouterLink: { props: ['to'], template: '<a :href="to"><slot /></a>' },
        WorkspaceNavigator: { template: '<button>Go to…</button>' },
        ThemeToggle: { template: '<div class="theme-toggle-stub" />' },
      },
    },
  })
}

describe('AppHeader', () => {
  it('shows a healthy Studio proxy indicator and links to the configured UI port', () => {
    const wrapper = mountHeader({ proxyStatus: { healthy: true, port: 2345 } })

    expect(wrapper.get('.status-light').classes()).toContain('status-light--online')
    expect(wrapper.get('a.proxy-link').attributes('href')).toBe('/proxy')
    expect(wrapper.text()).toContain('Studio')
  })

  it('keeps proxy status unknown until a health value arrives', async () => {
    const wrapper = mountHeader({ proxyStatus: null })

    expect(wrapper.get('.status-light').classes()).toContain('status-light--unknown')
    expect(wrapper.get('a.proxy-link').attributes('aria-label')).toContain('unknown')

    await wrapper.setProps({ proxyStatus: { healthy: false } })
    expect(wrapper.get('.status-light').classes()).toContain('status-light--offline')
  })

  it('opens Studio proxy management inside Studio', () => {
    const wrapper = mountHeader({ proxyStatus: { healthy: true, backend: 'studio' } })

    expect(wrapper.get('a.proxy-link').attributes('href')).toBe('/proxy')
    expect(wrapper.get('a.proxy-link').attributes('target')).toBeUndefined()
    expect(wrapper.text()).toContain('Studio')
  })
})

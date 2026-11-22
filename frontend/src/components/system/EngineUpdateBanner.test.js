import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import EngineUpdateBanner from './EngineUpdateBanner.vue'

function mountBanner(props = {}) {
  return mount(EngineUpdateBanner, {
    props,
    global: {
      directives: { tooltip: () => {} },
      stubs: {
        Button: {
          props: ['label', 'icon', 'severity', 'outlined', 'loading', 'ariaLabel'],
          emits: ['click'],
          template:
            '<button type="button" :data-label="label" :aria-label="ariaLabel" @click="$emit(\'click\')">{{ label }}</button>',
        },
      },
    },
  })
}

describe('EngineUpdateBanner', () => {
  it('announces an available update as a warning banner with the release link', () => {
    const wrapper = mountBanner({
      available: true,
      checked: true,
      latestVersion: 'b2000',
      currentVersion: 'b1999',
      linkUrl: 'https://example.test/releases/b2000',
      linkLabel: 'View release',
    })

    const banner = wrapper.get('.engine-update-banner')
    expect(banner.classes()).toContain('state-banner--warning')
    expect(banner.attributes('role')).toBe('status')
    expect(banner.text()).toContain('Update available')
    expect(banner.text()).toContain('b2000')
    const link = banner.get('a')
    expect(link.attributes('href')).toBe('https://example.test/releases/b2000')
    expect(link.text()).toBe('View release')
    expect(banner.get('[aria-label="Update to b2000"]').attributes('data-label')).toBe('Update')
  })

  it('emits update from the banner action', async () => {
    const wrapper = mountBanner({ available: true, checked: true, latestVersion: 'b2000' })

    await wrapper.get('.engine-update-banner button').trigger('click')
    expect(wrapper.emitted('update')).toHaveLength(1)
  })

  it('reports an up-to-date check as a success banner', () => {
    const wrapper = mountBanner({ available: false, checked: true, currentVersion: 'b2000' })

    const banner = wrapper.get('.engine-update-current')
    expect(banner.classes()).toContain('state-banner--success')
    expect(banner.text()).toContain('Up to date')
    expect(banner.text()).toContain('(b2000)')
  })

  it('renders nothing before a check has run', () => {
    const wrapper = mountBanner({ available: false, checked: false })

    expect(wrapper.find('.engine-update-banner').exists()).toBe(false)
    expect(wrapper.find('.engine-update-current').exists()).toBe(false)
  })
})

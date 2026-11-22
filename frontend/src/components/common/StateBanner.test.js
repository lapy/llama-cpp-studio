import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import StateBanner from './StateBanner.vue'

describe('StateBanner', () => {
  it('defaults to a warning status banner with an icon and message', () => {
    const wrapper = mount(StateBanner, { props: { message: 'Showing the last loaded library.' } })

    const banner = wrapper.get('.state-banner')
    expect(banner.classes()).toEqual(['state-banner', 'state-banner--warning'])
    expect(banner.attributes('role')).toBe('status')
    expect(banner.attributes('aria-live')).toBe('polite')
    expect(wrapper.get('.state-banner__icon').classes()).toContain('pi-exclamation-triangle')
    expect(wrapper.get('.state-banner__message').text()).toBe('Showing the last loaded library.')
  })

  it('maps an error to alert semantics without a redundant aria-live', () => {
    const wrapper = mount(StateBanner, {
      props: { severity: 'error', message: 'The model list could not be refreshed.' },
    })

    const banner = wrapper.get('.state-banner')
    expect(banner.classes()).toContain('state-banner--error')
    expect(banner.attributes('role')).toBe('alert')
    expect(banner.attributes('aria-live')).toBeUndefined()
    expect(wrapper.get('.state-banner__icon').classes()).toContain('pi-exclamation-circle')
  })

  it('keeps a caller-provided role and drops the icon when asked', () => {
    const wrapper = mount(StateBanner, {
      props: { severity: 'error', message: 'Save failed.', role: 'status', icon: '' },
    })

    const banner = wrapper.get('.state-banner')
    expect(banner.attributes('role')).toBe('status')
    expect(banner.attributes('aria-live')).toBe('polite')
    expect(wrapper.find('.state-banner__icon').exists()).toBe(false)
  })

  it('renders default slot content and action slots', () => {
    const wrapper = mount(StateBanner, {
      props: { severity: 'info' },
      slots: {
        default: '<span class="banner-copy">Draft stored in this session.</span>',
        actions: '<button>Restore draft</button>',
      },
    })

    expect(wrapper.get('.banner-copy').text()).toBe('Draft stored in this session.')
    expect(wrapper.get('.state-banner__actions button').text()).toBe('Restore draft')
    expect(wrapper.get('.state-banner').classes()).toContain('state-banner--info')
  })
})

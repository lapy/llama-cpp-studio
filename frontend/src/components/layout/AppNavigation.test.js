import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { reactive, nextTick } from 'vue'

const route = reactive({ name: 'search' })

vi.mock('vue-router', () => ({
  useRoute: () => route,
}))

import AppNavigation from './AppNavigation.vue'

const RouterLinkStub = {
  props: ['to'],
  template: '<a :href="to" v-bind="$attrs"><slot /></a>',
}

describe('AppNavigation', () => {
  it('renders core nav items and updates the active route state', async () => {
    const wrapper = mount(AppNavigation, {
      global: {
        stubs: {
          RouterLink: RouterLinkStub,
        },
      },
    })

    const links = wrapper.findAll('a')
    expect(links).toHaveLength(4)
    expect(wrapper.text()).toContain('Models')
    expect(wrapper.text()).toContain('Audio')
    expect(wrapper.text()).toContain('Search')
    expect(wrapper.text()).toContain('Engines')
    expect(links[2].attributes('aria-current')).toBe('page')
    expect(links[0].classes()).toContain('p-button-outlined')

    route.name = 'engines'
    await nextTick()

    const updatedLinks = wrapper.findAll('a')
    expect(updatedLinks[3].attributes('aria-current')).toBe('page')
    expect(updatedLinks[3].classes()).not.toContain('p-button-outlined')
  })
})

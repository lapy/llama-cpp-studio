import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createMemoryHistory, createRouter } from 'vue-router'
import SectionTabs from './SectionTabs.vue'

const TAB_ITEMS = [
  { id: 'overview', label: 'Overview', icon: 'pi-chart-line' },
  { id: 'requests', label: 'Requests', count: 4 },
  { id: 'settings', label: 'Settings' },
]

function mountTabs(props = {}, options = {}) {
  return mount(SectionTabs, {
    props: { items: TAB_ITEMS, ariaLabel: 'Proxy sections', ...props },
    ...options,
  })
}

describe('SectionTabs', () => {
  it('renders an ARIA tablist with roving tabindex and panel wiring', () => {
    const wrapper = mountTabs({ modelValue: 'requests', idPrefix: 'proxy' })

    const tablist = wrapper.get('[role="tablist"]')
    expect(tablist.attributes('aria-label')).toBe('Proxy sections')
    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs).toHaveLength(3)
    expect(tabs[1].attributes('aria-selected')).toBe('true')
    expect(tabs[1].attributes('tabindex')).toBe('0')
    expect(tabs[0].attributes('aria-selected')).toBe('false')
    expect(tabs[0].attributes('tabindex')).toBe('-1')
    expect(tabs[1].attributes('id')).toBe('proxy-tab-requests')
    expect(tabs[1].attributes('aria-controls')).toBe('proxy-panel-requests')
    expect(tabs[1].text()).toContain('Requests')
    expect(tabs[1].get('.section-tabs__count').text()).toBe('4')
  })

  it('omits ids and aria-controls when no panel prefix is configured', () => {
    const wrapper = mountTabs({ modelValue: 'overview' })

    const tabs = wrapper.findAll('[role="tab"]')
    expect(tabs[0].attributes('id')).toBeUndefined()
    expect(tabs[0].attributes('aria-controls')).toBeUndefined()
  })

  it('emits the selected id once when another tab is clicked', async () => {
    const wrapper = mountTabs({ modelValue: 'overview' })

    await wrapper.findAll('[role="tab"]')[2].trigger('click')
    expect(wrapper.emitted('update:modelValue')).toEqual([['settings']])

    await wrapper.findAll('[role="tab"]')[0].trigger('click')
    expect(wrapper.emitted('update:modelValue')).toEqual([['settings']])
  })

  it('does not emit for a disabled tab', async () => {
    const wrapper = mount(SectionTabs, {
      props: {
        items: [
          { id: 'overview', label: 'Overview' },
          { id: 'requests', label: 'Requests', disabled: true },
        ],
        modelValue: 'overview',
        ariaLabel: 'Proxy sections',
      },
    })

    const disabled = wrapper.findAll('[role="tab"]')[1]
    expect(disabled.attributes('disabled')).toBeDefined()
    await disabled.trigger('click')
    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
  })

  it('moves selection and focus with arrow, Home and End keys', async () => {
    const wrapper = mount(
      {
        components: { SectionTabs },
        data: () => ({ active: 'overview', items: TAB_ITEMS }),
        template:
          '<SectionTabs :items="items" v-model="active" aria-label="Proxy sections" id-prefix="proxy" />',
      },
      { attachTo: document.body },
    )

    const tabs = wrapper.findAll('[role="tab"]')
    tabs[0].element.focus()

    await tabs[0].trigger('keydown', { key: 'ArrowRight' })
    expect(wrapper.get('[role="tab"][aria-selected="true"]').text()).toContain('Requests')
    expect(document.activeElement).toBe(tabs[1].element)

    await tabs[1].trigger('keydown', { key: 'End' })
    expect(wrapper.get('[role="tab"][aria-selected="true"]').text()).toContain('Settings')
    expect(document.activeElement).toBe(tabs[2].element)

    await tabs[2].trigger('keydown', { key: 'Home' })
    expect(wrapper.get('[role="tab"][aria-selected="true"]').text()).toContain('Overview')
    expect(document.activeElement).toBe(tabs[0].element)

    wrapper.unmount()
  })

  it('renders router links with aria-current in links mode', async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [{ path: '/:pathMatch(.*)*', component: { template: '<div />' } }],
    })
    await router.push('/system')
    await router.isReady()

    const wrapper = mount(SectionTabs, {
      props: {
        items: [
          { id: 'hardware', label: 'Hardware', to: '/system#hardware' },
          { id: 'diagnostics', label: 'Diagnostics', to: '/system' },
        ],
        modelValue: 'diagnostics',
        ariaLabel: 'System sections',
      },
      global: { plugins: [router] },
    })

    expect(wrapper.find('[role="tablist"]').exists()).toBe(false)
    const links = wrapper.findAll('a.section-tabs__tab')
    expect(links).toHaveLength(2)
    expect(links[0].attributes('href')).toBe('/system#hardware')
    expect(links[0].attributes('aria-current')).toBeUndefined()
    expect(links[1].attributes('href')).toBe('/system')
    expect(links[1].attributes('aria-current')).toBe('page')
  })

  it('applies the pills variant and responsive grid modifier', () => {
    const wrapper = mountTabs({ modelValue: 'overview', variant: 'pills', responsiveGrid: true })

    const tablist = wrapper.get('[role="tablist"]')
    expect(tablist.classes()).toContain('section-tabs--pills')
    expect(tablist.classes()).toContain('section-tabs--responsive-grid')
  })
})

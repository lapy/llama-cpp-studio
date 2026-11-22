import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import PageToolbar from './PageToolbar.vue'

describe('PageToolbar', () => {
  it('renders search, filter group, summary and actions with an accessible group name', () => {
    const wrapper = mount(PageToolbar, {
      props: { label: 'Library filters' },
      slots: {
        search: '<input class="search-input" aria-label="Search library" />',
        default: '<select class="filter-select" aria-label="Status" />',
        summary: '<span class="result-count">2 models</span>',
        actions: '<button>Refresh</button>',
      },
    })

    const toolbar = wrapper.get('.page-toolbar')
    expect(toolbar.attributes('role')).toBe('group')
    expect(toolbar.attributes('aria-label')).toBe('Library filters')
    expect(wrapper.get('.page-toolbar__search .search-input').exists()).toBe(true)
    expect(wrapper.get('.page-toolbar__group .filter-select').exists()).toBe(true)
    expect(wrapper.get('.page-toolbar__summary .result-count').text()).toBe('2 models')
    expect(wrapper.get('.page-toolbar__actions button').text()).toBe('Refresh')
  })

  it('omits empty regions and supports the plain variant', () => {
    const wrapper = mount(PageToolbar, {
      props: { plain: true },
      slots: { default: '<span class="only-filter">Status</span>' },
    })

    const toolbar = wrapper.get('.page-toolbar')
    expect(toolbar.classes()).toContain('page-toolbar--plain')
    expect(toolbar.attributes('role')).toBeUndefined()
    expect(toolbar.attributes('aria-label')).toBeUndefined()
    expect(wrapper.find('.page-toolbar__search').exists()).toBe(false)
    expect(wrapper.find('.page-toolbar__summary').exists()).toBe(false)
    expect(wrapper.find('.page-toolbar__actions').exists()).toBe(false)
    expect(wrapper.get('.page-toolbar__group .only-filter').exists()).toBe(true)
  })
})

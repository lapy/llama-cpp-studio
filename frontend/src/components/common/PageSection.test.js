import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import PageSection from './PageSection.vue'

describe('PageSection', () => {
  it('renders a labelled section with heading, description, meta, actions and body', () => {
    const wrapper = mount(PageSection, {
      props: { title: 'Storage', description: 'Disk use Studio owns.' },
      slots: {
        default: '<p class="body-copy">Body</p>',
        meta: '<span class="meta-tag">3 categories</span>',
        actions: '<button>Clean up</button>',
      },
    })

    const section = wrapper.get('section.page-section')
    const heading = wrapper.get('h2.page-section__title')
    expect(heading.text()).toBe('Storage')
    expect(section.attributes('aria-labelledby')).toBe(heading.attributes('id'))
    expect(wrapper.get('.page-section__description').text()).toBe('Disk use Studio owns.')
    expect(wrapper.get('.page-section__meta .meta-tag').text()).toBe('3 categories')
    expect(wrapper.get('.page-section__actions button').text()).toBe('Clean up')
    expect(wrapper.get('.page-section__body .body-copy').text()).toBe('Body')
  })

  it('honours heading level, anchor id and variant', () => {
    const wrapper = mount(PageSection, {
      props: { title: 'Hardware', headingTag: 'h3', id: 'hardware', variant: 'compact' },
    })

    const section = wrapper.get('section')
    expect(section.attributes('id')).toBe('hardware')
    expect(section.classes()).toContain('page-section--compact')
    const heading = wrapper.get('h3.page-section__title')
    expect(heading.attributes('id')).toBe('hardware-title')
    expect(section.attributes('aria-labelledby')).toBe('hardware-title')
  })

  it('omits the header entirely when no heading content is supplied', () => {
    const wrapper = mount(PageSection, { slots: { default: '<p>Only body</p>' } })

    expect(wrapper.find('.page-section__header').exists()).toBe(false)
    expect(wrapper.find('section').attributes('aria-labelledby')).toBeUndefined()
    expect(wrapper.get('.page-section__body').text()).toBe('Only body')
  })

  it('keeps a custom title slot out of the generated aria-labelledby contract', () => {
    const wrapper = mount(PageSection, {
      slots: { title: '<h2 class="custom-title">Request log</h2>' },
    })

    expect(wrapper.get('.custom-title').text()).toBe('Request log')
    expect(wrapper.get('section').attributes('aria-labelledby')).toBeUndefined()
  })

  it('applies the plain variant without card chrome', () => {
    const wrapper = mount(PageSection, {
      props: { title: 'Runtime engines', variant: 'plain' },
    })

    expect(wrapper.get('section').classes()).toContain('page-section--plain')
  })
})

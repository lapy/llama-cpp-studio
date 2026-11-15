import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import LibrarySavedFilters from './LibrarySavedFilters.vue'
import { LIBRARY_PRESETS_KEY, loadPresets } from '@/utils/libraryFilterPresets'

const FILTERS = {
  q: '  llama  ',
  status: 'attention',
  engine: 'llama.cpp',
  task: 'all',
  sort: 'recent',
}

function mountPresets(filters = FILTERS) {
  return mount(LibrarySavedFilters, { props: { filters } })
}

describe('LibrarySavedFilters', () => {
  beforeEach(() => {
    localStorage.removeItem(LIBRARY_PRESETS_KEY)
  })

  it('saves the current filters under a name and lists the normalized summary', async () => {
    const wrapper = mountPresets()

    expect(wrapper.text()).toContain('No saved presets yet')
    await wrapper.get('#library-preset-name').setValue('Needs attention')
    await wrapper.get('form').trigger('submit')

    const stored = loadPresets()
    expect(stored).toHaveLength(1)
    expect(stored[0]).toMatchObject({
      name: 'Needs attention',
      filters: {
        q: 'llama',
        status: 'attention',
        engine: 'llama.cpp',
        task: 'all',
        sort: 'recent',
      },
    })
    expect(wrapper.text()).toContain('Saved “Needs attention”')
    expect(wrapper.get('.library-presets__summary').text()).toContain('Status: Needs attention')
    expect(wrapper.get('.library-presets__summary').text()).toContain('Sort: Recent use')
    expect(wrapper.get('#library-preset-name').element.value).toBe('')
  })

  it('keeps save disabled until a name is entered and reports duplicate names', async () => {
    const wrapper = mountPresets()

    const saveButton = wrapper.get('button[type="submit"]')
    expect(saveButton.attributes('disabled')).toBeDefined()

    await wrapper.get('#library-preset-name').setValue('Mine')
    await wrapper.get('form').trigger('submit')
    await wrapper.get('#library-preset-name').setValue('mine')
    await wrapper.get('form').trigger('submit')

    expect(wrapper.get('[role="alert"]').text()).toContain('already exists')
    expect(loadPresets()).toHaveLength(1)
  })

  it('applies a stored preset through the apply event', async () => {
    localStorage.setItem(
      LIBRARY_PRESETS_KEY,
      JSON.stringify({
        version: 1,
        presets: [
          {
            id: 'preset-1',
            name: 'Running only',
            filters: { q: 'llama', status: 'running', engine: 'all', task: 'all', sort: 'size' },
            createdAt: '2026-11-15T00:00:00.000Z',
          },
        ],
      }),
    )
    const wrapper = mountPresets()
    wrapper.get('details').element.open = true

    await wrapper.get('button[aria-label="Apply preset Running only"]').trigger('click')

    expect(wrapper.emitted('apply')).toEqual([
      [{ q: 'llama', status: 'running', engine: 'all', task: 'all', sort: 'size' }],
    ])
    expect(wrapper.get('[role="status"]').text()).toContain('Applied “Running only”')
    expect(wrapper.get('details').element.open).toBe(false)
  })

  it('removes a deleted preset from storage and the list', async () => {
    const wrapper = mountPresets()
    await wrapper.get('#library-preset-name').setValue('Temporary')
    await wrapper.get('form').trigger('submit')

    await wrapper.get('button[aria-label="Delete preset Temporary"]').trigger('click')

    expect(loadPresets()).toEqual([])
    expect(wrapper.text()).toContain('Deleted “Temporary”')
    expect(wrapper.text()).toContain('No saved presets yet')
  })

  it('shows an empty list for unreadable local data instead of failing', () => {
    localStorage.setItem(LIBRARY_PRESETS_KEY, '{not json')

    const wrapper = mountPresets()

    expect(wrapper.text()).toContain('No saved presets yet')
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
  })
})

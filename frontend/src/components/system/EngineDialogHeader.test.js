import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import EngineDialogHeader from './EngineDialogHeader.vue'

function mountHeader() {
  return mount(EngineDialogHeader, {
    props: { title: 'audio.cpp' },
    slots: {
      leading: '<span class="engine-mark">A</span>',
      tags: '<span class="engine-dialog-tag-clip">linux-cpu-local</span>',
      actions: '<button type="button" aria-label="Build settings">settings</button>',
      more: `
        <button type="button" aria-label="Reload versions">reload</button>
        <button type="button" aria-label="Rescan audio.cpp capabilities">rescan</button>
      `,
    },
    global: {
      stubs: {
        Button: {
          props: ['icon', 'ariaLabel'],
          template: '<button type="button" :aria-label="ariaLabel || \'More actions\'"><slot /></button>',
        },
      },
    },
  })
}

describe('EngineDialogHeader', () => {
  it('keeps the title and primary action in the header row', () => {
    const wrapper = mountHeader()
    expect(wrapper.get('.engine-dialog-header__title').text()).toBe('audio.cpp')
    expect(wrapper.get('[aria-label="Build settings"]').exists()).toBe(true)
    expect(wrapper.get('[aria-label="More actions"]').exists()).toBe(true)
    expect(wrapper.get('[aria-label="Reload versions"]').exists()).toBe(true)
  })

  it('opens and closes the overflow menu', async () => {
    const wrapper = mountHeader()
    const more = wrapper.get('.engine-dialog-header__more')
    expect(more.classes()).not.toContain('is-open')

    await wrapper.get('[aria-label="More actions"]').trigger('click')
    expect(more.classes()).toContain('is-open')

    await wrapper.get('[aria-label="Reload versions"]').trigger('click')
    expect(more.classes()).not.toContain('is-open')
  })

  it('styles the overflow menu with a themed solid background', () => {
    const src = readFileSync(
      resolve(dirname(fileURLToPath(import.meta.url)), './EngineDialogHeader.vue'),
      'utf8',
    )
    expect(src).toContain('background: var(--bg-secondary)')
    expect(src).not.toMatch(/var\(--bg-card, var\(--surface-0\)\)/)
  })
})

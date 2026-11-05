import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { ref, nextTick } from 'vue'

const isDark = ref(false)
const toggleTheme = vi.fn()

vi.mock('@/composables/useTheme', () => ({
  useTheme: () => ({ isDark, toggleTheme }),
}))

import ThemeToggle from './ThemeToggle.vue'

const ButtonStub = {
  props: ['icon', 'severity', 'text', 'title'],
  emits: ['click'],
  template: '<button :data-icon="icon" :title="title" :class="$attrs.class" @click="$emit(`click`)"><slot /></button>',
}

describe('ThemeToggle', () => {
  beforeEach(() => {
    isDark.value = false
    toggleTheme.mockReset()
  })

  it('renders the correct icon/title and toggles the theme', async () => {
    const wrapper = mount(ThemeToggle, {
      global: {
        stubs: {
          Button: ButtonStub,
        },
      },
    })

    expect(wrapper.get('button').attributes('data-icon')).toBe('pi pi-moon')
    expect(wrapper.get('button').attributes('title')).toBe('Switch to Dark Mode')
    expect(wrapper.get('button').classes()).not.toContain('is-dark')

    await wrapper.get('button').trigger('click')
    expect(toggleTheme).toHaveBeenCalledTimes(1)

    isDark.value = true
    await nextTick()

    expect(wrapper.get('button').attributes('data-icon')).toBe('pi pi-sun')
    expect(wrapper.get('button').attributes('title')).toBe('Switch to Light Mode')
    expect(wrapper.get('button').classes()).toContain('is-dark')
  })
})

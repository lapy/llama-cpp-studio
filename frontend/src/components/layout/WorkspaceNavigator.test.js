import { mount, flushPromises } from '@vue/test-utils'
import { describe, it, expect, vi } from 'vitest'
const push = vi.fn()
vi.mock('vue-router', () => ({ useRouter: () => ({ push }) }))
import WorkspaceNavigator from './WorkspaceNavigator.vue'
function mountNavigator() {
  return mount(WorkspaceNavigator, {
    global: {
      stubs: {
        Dialog: {
          props: ['visible'],
          emits: ['update:visible'],
          template: '<div v-if="visible" role="dialog"><slot /></div>',
        },
      },
    },
  })
}
describe('WorkspaceNavigator', () => {
  it('opens with the workspace shortcut, searches tools, and navigates with Enter', async () => {
    const wrapper = mountNavigator()
    window.dispatchEvent(
      new window.KeyboardEvent('keydown', { key: 'k', ctrlKey: true, cancelable: true }),
    )
    await flushPromises()
    const input = wrapper.get('input')
    await input.setValue('backup')
    expect(wrapper.findAll('[role="option"]')).toHaveLength(1)
    expect(wrapper.text()).toContain('Backup & restore')
    await input.trigger('keydown', { key: 'Enter' })
    expect(push).toHaveBeenCalledWith('/system#config-backup')
    expect(wrapper.find('[role="dialog"]').exists()).toBe(false)
    wrapper.unmount()
  })
  it('supports arrow keys and handles an empty search without navigating', async () => {
    push.mockClear()
    const wrapper = mountNavigator()
    await wrapper.get('.workspace-search').trigger('click')
    const input = wrapper.get('input')
    await input.trigger('keydown', { key: 'ArrowDown' })
    expect(wrapper.findAll('[role="option"]')[1].attributes('aria-selected')).toBe('true')
    await input.setValue('nothing matches this')
    await input.trigger('keydown', { key: 'ArrowDown' })
    await input.trigger('keydown', { key: 'Enter' })
    expect(push).not.toHaveBeenCalled()
    expect(wrapper.text()).toContain('No destinations match')
    wrapper.unmount()
    window.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'k', ctrlKey: true }))
  })
})
